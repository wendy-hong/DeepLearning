import tensorflow as tf

labels = [[0, 0, 1], [0, 1, 0]]
logits = [[2, 0.5, 6], [0.1, 0, 3]]
logits_softmax = tf.nn.softmax(logits)
logits_softmax2 = tf.nn.softmax(logits_softmax)

result1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)  # softmax+cross_entropy
result2 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits_softmax)  # softmax*2+cross_entropy
result3 = -tf.reduce_sum(labels * tf.log(logits_softmax), 1)  # softmax+cross_entropy

labels2 = [[0.4, 0.1, 0.5], [0.3, 0.6, 0.1]]
result4 = tf.nn.softmax_cross_entropy_with_logits(labels=labels2, logits=logits)

# Sparse
labels3 = [2, 1]
result5 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels3, logits=logits)

loss1 = tf.reduce_mean(result1)
loss2 = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(logits_softmax), 1))

with tf.Session() as sess:
    print("softmax=", sess.run(logits_softmax))
    print("softmax2=", sess.run(logits_softmax2))
    print("rel1=", sess.run(result1), "Correct!")
    print("rel2=", sess.run(result2), "Wrong!")
    print("rel3=", sess.run(result3), "Correct!")
    print("rel4=", sess.run(result4), "Non-one_hot, not obvious")
    print("rel5=", sess.run(result5), "Sparse")
    print("loss1=", sess.run(loss1))
    print("loss2=", sess.run(loss2))
