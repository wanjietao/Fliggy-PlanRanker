# coding: utf-8
import tensorflow as tf
from tensorflow.contrib import layers

from xdeepctr.layers.core import DenseLayer
from xdeepctr.utils import get_emb_list, combined_dnn_input
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor

LINEAR_SCOPE = 'Wide'
DNN_SCOPE = 'Deep'
DNN_LEARNING_RATE = 0.001
LINEAR_LEARNING_RATE = 0.005
# wide : _SparseColumnHashed
# deep : _EmbeddingColumn、_RealValuedColumn

def model_fn(features, labels, params, mode):
    training_mode = mode
    #  模型参数
    model_conf = params["algo_conf"]
    model_conf["model_hyperparameter"] = dict(model_conf["model_hyperparameter"], chief_hooks=None)
    #  优化器参数
    optimizer_conf = params["optimizer"]
    feature_columns = params["feature_columns"]
    linear_input = True if feature_columns.get("wide", None) else False
    dnn_input = True if feature_columns.get("deep", None) else False
    task_name = get_model_hyperparameter(model_conf, "task")
    chief_hooks = get_model_hyperparameter(model_conf, "chief_hooks")
    logits = inference(features, training_mode, model_conf, feature_columns)
    score = tf.sigmoid(logits) if task_name == "binary" else logits
    tf.identity(score, name="rank_predict")
    try:
        tf.contrib.layers.add_tensor_to_collection(
            ops.GraphKeys.RANK_SERVICE_OUTPUT, 'score', score)
    except:
        pass
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            # 'class_ids': predicted_classes[:, tf.newaxis],
            'score': score,
            'logits': logits,
            'label': features.get("labels"),
            # 'linked_id': features.get("linked_id"),
            'ids': features.get("ids")
        }
        extra_output = get_output_tensors(features, params)
        predictions = dict(predictions, **extra_output)
        print(predictions)
        output_score = tf.placeholder(
            tf.float32, 1, name="output-tensor-scores")
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs={
            "score": tf.estimator.export.ClassificationOutput(scores=output_score)})
    metrics = get_metrics(logits, labels, task_name)
    loss = get_loss(logits, labels, task_name)
    train_op = get_train_op(loss, linear_input, dnn_input, optimizer_conf)
    train_log_metrics = {
        "loss": loss,
        "metrics": metrics["auc"][1],
        "step": tf.train.get_global_step()
    }
    train_log_hook = tf.train.LoggingTensorHook(train_log_metrics, every_n_iter=100)
    return tf.estimator.EstimatorSpec(mode=mode, predictions=None, loss=loss, train_op=train_op,
                                      eval_metric_ops=metrics,
                                      training_chief_hooks=chief_hooks,
                                      training_hooks=[train_log_hook],
                                      evaluation_hooks=None)


def inference(features, training_mode, model_conf, feature_columns):
    linear_feature_columns = feature_columns.get("wide", None)
    dnn_feature_columns = feature_columns.get("deep", None)
    dnn_hidden_units = get_model_hyperparameter(model_conf, "dnn_hidden_units", [256, 128, 64])
    dnn_activation = get_model_hyperparameter(model_conf, "dnn_activation", "relu")
    dnn_dropout = get_model_hyperparameter(model_conf, "dnn_dropout", 0.0)
    dnn_use_bn = get_model_hyperparameter(model_conf, "dnn_use_bn", False)
    l2_reg_embedding = get_model_hyperparameter(model_conf, "l2_reg_embedding", 0.0)
    l2_reg_linear = get_model_hyperparameter(model_conf, "l2_reg_linear", 0.0)
    l2_reg_dnn = get_model_hyperparameter(model_conf, "l2_reg_dnn", 0.0)
    seed = get_model_hyperparameter(model_conf, "seed", 1024)
    train_flag = is_training(training_mode)
    with tf.variable_scope(LINEAR_SCOPE, partitioner=partitioner()):
        linear_logit = get_linear_logit(features, linear_feature_columns, l2_reg_linear)
    with tf.variable_scope(DNN_SCOPE, partitioner=partitioner()):

        sparse_emb_list, dense_value_list = get_emb_list(
            features, dnn_feature_columns, l2_reg_embedding)

        dnn_input = combined_dnn_input(sparse_emb_list, dense_value_list)

        for hidden_unit in dnn_hidden_units:
            hidden_output = DenseLayer(hidden_unit=hidden_unit, activation=dnn_activation, l2_reg=l2_reg_dnn,
                                       dropout_rate=dnn_dropout,
                                       use_bn=dnn_use_bn, seed=seed)(dnn_input, training=train_flag)
            dnn_input = hidden_output
        deep_logit = DenseLayer(hidden_unit=1, activation=None, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                use_bn=False, seed=seed)(dnn_input, training=train_flag)
    print("monotonicity_time shape: ", features["monotonicity_time"].get_shape())
    print("monotonicity_price shape: ", features["monotonicity_price"].get_shape())
    print("monotonicity_time shape1: ", features["monotonicity_time"].shape)
    print("monotonicity_price shape1: ", features["monotonicity_price"].shape)
    with tf.variable_scope("monotonicity_time_dnn"):
        first_layer = get_monotonicity_dnn(features["monotonicity_time"], 64, "first_layer", activation="relu")
        second_layer = get_monotonicity_dnn(first_layer, 32, "first_layer", activation="relu")
        monotonicity_time_logits = get_monotonicity_dnn(second_layer, 1, "ouptut")
    with tf.variable_scope("monotonicity_price_dnn"):
        first_layer = get_monotonicity_dnn(features["monotonicity_price"], 64, "first_layer", activation="relu")
        second_layer = get_monotonicity_dnn(first_layer, 32, "first_layer", activation="relu")
        monotonicity_price_logits = get_monotonicity_dnn(second_layer, 1, "ouptut")
    if len(dnn_hidden_units) == 0:  # only linear
        final_logit = linear_logit
    else:
        tf.contrib.layers.summarize_tensor(deep_logit, tag='dnn_logit')
        final_logit = linear_logit + deep_logit  # linear + deep
    logits = final_logit + 0.8 * monotonicity_time_logits + 0.2 * monotonicity_price_logits
    return logits


def get_model_hyperparameter(model_conf, name, default_value=None):
    if name in model_conf["model_hyperparameter"]:
        return model_conf["model_hyperparameter"][name]
    elif default_value is not None:
        return default_value
    else:
        raise ValueError(name + "not in model_hyperparameter")


def is_training(training_mode):
    if training_mode is not None:
        if training_mode == tf.estimator.ModeKeys.TRAIN:
            return True
        else:
            return tf.placeholder_with_default(False, [], name='training')
    try:
        training = tf.get_default_graph().get_tensor_by_name("training:0")
    except KeyError as e:
        training = tf.placeholder(tf.bool, name="training")
    return training


def partitioner():
    return None


def get_linear_logit(features, feature_columns, l2_reg_linear=0.0, units=1, sparse_combiner='sum'):

    if feature_columns is None or len(feature_columns) == 0:
        return tf.constant(0.0)

    linear_logits, collections_linear_weights, linear_bias = \
        layers.weighted_sum_from_feature_columns(
            columns_to_tensors=features,
            feature_columns=feature_columns,
            num_outputs=1,
            weight_collections=None,
            scope=None)  # wide_scope
    linear_logit = linear_logits + linear_bias
    weights_list = list(map(lambda x: x[0], collections_linear_weights.values()))
    tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(float(l2_reg_linear)),
                                           weights_list=weights_list)
    return linear_logit
    # try:
    #     cols_to_vars = {}
    #     linear_logit = tf.feature_column.linear_model(
    #         features, feature_columns, units=units,
    #         sparse_combiner=sparse_combiner,
    #         weight_collections=None,
    #         trainable=True,
    #         cols_to_vars=cols_to_vars)
    #
    #     bias = cols_to_vars.pop('bias')
    #     tf.contrib.layers.summarize_tensor(bias, tag='bias')
    #     tf.contrib.layers.summarize_tensor(_compute_fraction_of_zero(
    #         cols_to_vars), tag='fraction_of_zero_weights')
    # except:  # PAI TF
    #     linear_logit = tf.feature_column.linear_model(
    #         features, feature_columns, units=units,
    #         sparse_combiner=sparse_combiner,
    #         weight_collections=None,
    #         trainable=True, )

    # return linear_logit


def get_output_tensors(features, params):
    extra_output = {}
    if params["fg"].mc.has_block("output"):
        for feat in params["fg"].mc.get_column_names_by_block_name("output"):
            output_tensor = features[feat]
            if isinstance(output_tensor, sparse_tensor.SparseTensor):

                if output_tensor.dtype == tf.string:
                    default_value = "0"
                else:
                    default_value = 0

                extra_output[feat] = tf.reshape(
                    tf.sparse_tensor_to_dense(output_tensor, default_value=default_value),
                    [-1])
            else:
                extra_output[feat] = features[feat]
    return extra_output


def get_loss(logits, labels, task):
    logits = tf.reshape(logits, [-1, 1])
    labels = tf.reshape(labels, [-1, 1])
    with tf.name_scope("loss_op"):
        if task == "binary":
            binary_cross_entropy = tf.losses.sigmoid_cross_entropy(
                labels, logits)
            loss_op = binary_cross_entropy
        else:
            mean_squared_error = tf.losses.mean_squared_error(labels, logits)
            loss_op = mean_squared_error

        loss_op += tf.losses.get_regularization_loss()
    return loss_op


def get_metrics(logits, labels, task):
    labels = tf.reshape(labels, [-1, 1])
    logits = tf.reshape(logits, [-1, 1])
    if task== "binary":
        score = tf.sigmoid(logits)
        positive_score = tf.boolean_mask(
            score, tf.cast(labels, tf.bool))  # score * labels
        negative_score = tf.boolean_mask(
            score, tf.cast(1 - labels, tf.bool))
        tf.contrib.layers.summarize_tensor(
            positive_score, tag='score/positive')
        tf.contrib.layers.summarize_tensor(
            negative_score, tag='score/negative')
        binary_cross_entropy = tf.losses.sigmoid_cross_entropy(
            labels, logits)
        auc_roc = tf.metrics.auc(tf.reshape(
            labels, [-1]), score, name='auc_roc_op')
        # auc_pr = tf.metrics.auc(tf.reshape(
        #    labels, [-1]), score, curve='PR', name='auc_pr_op')
        score_metric = tf.metrics.mean(score, name='score_op')
        # positive_score
        metrics = {'auc': auc_roc,  # "auc_precision_recall": auc_pr,
                   "binary_crossentropy": tf.metrics.mean(binary_cross_entropy),
                   }

        tf.summary.scalar('auc', auc_roc[1])
        # tf.summary.scalar('auc_precision_recall', auc_pr[1])
        tf.summary.scalar("binary_crossentropy", binary_cross_entropy)
    else:
        score = logits

        mean_squared_error = tf.losses.mean_squared_error(labels, logits)

        score_metric = tf.metrics.mean(score, name='score_op')

        labels = tf.cast(labels, tf.float32)
        mse = tf.metrics.mean_squared_error(tf.reshape(
            labels, [-1]), score, name='rmse_op')
        metrics = {'mean_squared_error': mse,
                   }

        tf.summary.scalar("mean_squared_error", mean_squared_error)

    score_max = tf.reduce_max(score, axis=0)
    score_min = tf.reduce_min(score, axis=0)
    score_max_metric = tf.metrics.mean(score_max, name='score_max_op')
    score_min_metric = tf.metrics.mean(score_min, name='score_min_op')

    tf.summary.scalar('score_mean', score_metric[1])
    tf.summary.scalar('score_max', score_max_metric[1])
    tf.summary.scalar('score_min', score_min_metric[1])
    metrics['score_mean'] = score_metric
    metrics['score_max'] = score_max_metric
    metrics['score_min'] = score_min_metric

    tf.contrib.layers.summarize_tensor(score, tag='score/all')

    return metrics


def get_train_op(loss, linear_input, dnn_input, optimizer_conf):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    global_step = tf.train.get_global_step()
    if dnn_input:
        dnn_optimizer = find_optimizer(DNN_SCOPE, optimizer_conf)
        with tf.control_dependencies(update_ops):
            train_var_list = tf.trainable_variables(DNN_SCOPE)

            # deep_train_op_loss = tf.contrib.training.create_train_op(loss, deep_opt,
            #                                                          global_step=tf.train.get_global_step(),
            #                                                          variables_to_train=train_var_list,
            #                                                          summarize_gradients=True)

            deep_train_op = dnn_optimizer.minimize(
                loss, global_step=global_step, var_list=train_var_list)
            # #tf.contrib.layers.optimize_loss(loss, global_step=tf.train.get_global_step(), variables=train_var_list)
            # deep_opt.minimize(loss, global_step=tf.train.get_global_step(),
            # var_list=train_var_list)
    if linear_input:  #
        linear_optimizer = find_optimizer(LINEAR_SCOPE, optimizer_conf)
        # wide_train_op_loss = tf.contrib.training.create_train_op(loss, wide_opt,
        #                                                          global_step=tf.train.get_global_step(),
        #                                                          variables_to_train=tf.trainable_variables(
        #                                                              _LINEAR_LEARNING_RATE),
        #                                                          summarize_gradients=True)  # 这里只是对wide进行参数更新，但是返回的仍然是总的loss
        # tf.contrib.layers.optimize_loss(loss, global_step=tf.train.get_global_step(), tf.trainable_variables('wide'))
        wide_train_op = linear_optimizer.minimize(loss, global_step=global_step,
                                                  var_list=tf.trainable_variables(LINEAR_SCOPE))
    if linear_input and dnn_input:
        train_op = tf.group(deep_train_op, wide_train_op)  # optimizer
    elif dnn_input:
        train_op = deep_train_op
    else:
        train_op = wide_train_op
    return train_op


def find_optimizer(scope, optimizer_conf):
    global_opt = None
    for opt_name, opt_kw in optimizer_conf.items():
        if opt_kw["scope"] == scope or opt_kw["scope"] == "Global":
            lr = opt_kw.get("learning_rate", DNN_LEARNING_RATE)
            # lr = tf.train.exponential_decay(lr,tf.train.get_global_step(),)
            use_locking = opt_kw.get("use_locking", False)
            if not isinstance(use_locking, bool):
                if use_locking == "True":
                    use_locking = True
                else:
                    use_locking = False
            if opt_name == "Adagrad":
                opt = tf.train.AdagradOptimizer(learning_rate=lr,
                                                initial_accumulator_value=opt_kw.get("initial_accumulator_value",
                                                                                     0.1), use_locking=use_locking)
            elif opt_name == "Adam":
                opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=opt_kw.get("beta1", 0.9),
                                             beta2=opt_kw.get("beta2", 0.999), epsilon=opt_kw.get("epsilon", 1e-8),
                                             use_locking=use_locking)
            elif opt_name == "Adadelta":
                opt = tf.train.AdadeltaOptimizer(learning_rate=lr, rho=opt_kw.get("rho", 0.95),
                                                 epsilon=opt_kw.get("epsilon", 1e-8), use_locking=use_locking)
            elif opt_name == "Momentum":
                opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=opt_kw.get("momentum"),
                                                 use_locking=use_locking,
                                                 use_nesterov=opt_kw.get("use_nesterov", False))
            elif opt_name == "Ftrl":
                lr = opt_kw.get("learning_rate", LINEAR_LEARNING_RATE)
                opt = tf.train.FtrlOptimizer(learning_rate=lr,
                                             learning_rate_power=opt_kw.get("learning_rate_power", -0.5),
                                             initial_accumulator_value=opt_kw.get("initial_accumulator_value", 0.1),
                                             l1_regularization_strength=opt_kw.get("l1_regularization_strength",
                                                                                   0.0),
                                             l2_regularization_strength=opt_kw.get("l2_regularization_strength",
                                                                                   0.0),
                                             use_locking=use_locking,
                                             l2_shrinkage_regularization_strength=opt_kw.get(
                                                 "l2_shrinkage_regularization_strength", 0.0)
                                             )
            else:
                raise ValueError("optimizer config error! Now supports Adagrad,Adam,Adadelta,Momentum,Ftrl")

            if opt_kw["scope"] == scope:
                return opt
            else:
                global_opt = opt
    if global_opt is None:
        raise ValueError("optimizer config error! scope must have Wide,Deep or Global")
    return global_opt

def get_monotonicity_dnn(x, output_shape, name, activation = None):
    with tf.name_scope(name):
        n_inputs = int(x.get_shape()[1])
        # 设置权重和偏置，权重为随机选取的正太分布，偏置设为0
        W = tf.Variable(tf.random_uniform([n_inputs, output_shape], -1.0, 1.0), name="weights")
        b = tf.Variable(tf.zeros([output_shape]), name="biases")
        # 计算待输出结果
        z = tf.matmul(x, W * W) + b # squared weights
        # 是否要激活函数计算
        if activation == "relu":
            return tf.nn.relu(z)
        else:
            return z
