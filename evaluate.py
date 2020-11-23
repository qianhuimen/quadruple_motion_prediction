
"""Simple code for evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import numpy as np
import tensorflow as tf
import forward_kinematics
import data_utils

tf.app.flags.DEFINE_string("action","all", "The action to train on. all means all the actions, all_periodic means walking, eating and smoking")
FLAGS = tf.app.flags.FLAGS

def evaluate():
  """Evaluations for srnn's seeds"""

  actions = define_actions( FLAGS.action )
  seq_length_out = 25
  # Load all the data
  parent, offset, rotInd, expmapInd = forward_kinematics._some_variables()


  with h5py.File('samples.h5', 'r') as h5f:
    # Predict and save for each action
    for action in actions:

      xyz_gt = []
      xyz_pred = []

      # Compute and save the errors here
      mean_errors=[]
      for i in np.arange(8):
          srnn_pred = h5f['expmap/preds/' + action + '_' + str(i)][:seq_length_out, :]
          srnn_gts = h5f['expmap/gt/' + action + '_' + str(i)][:seq_length_out, :]

          rotation_pred = forward_kinematics.revert_coordinate_space(srnn_pred, np.eye(3), np.zeros(3))
          rotation_gt = forward_kinematics.revert_coordinate_space(srnn_gts, np.eye(3), np.zeros(3))
          xyz_gt_tmp, xyz_pred_tmp = np.zeros((seq_length_out, 96)), np.zeros((seq_length_out, 96))

          # Compute 3d points for each frame
          for j in range(seq_length_out):
              xyz_gt_tmp[j, :] = forward_kinematics.fkl(rotation_gt[j, :], parent, offset, rotInd, expmapInd)
          for j in range(seq_length_out):
              xyz_pred_tmp[j, :] = forward_kinematics.fkl(rotation_pred[j, :], parent, offset, rotInd, expmapInd)

          xyz_gt.append(xyz_gt_tmp)
          xyz_pred.append(xyz_pred_tmp)

          # change back to euler to evaluate MAE
          for j in np.arange(srnn_pred.shape[0]):
              for k in np.arange(3, 97, 3):
                  srnn_gts[j, k:k + 3] = data_utils.rotmat2euler(
                      data_utils.expmap2rotmat(srnn_gts[j, k:k + 3]))
                  srnn_pred[j, k:k + 3] = data_utils.rotmat2euler(
                      data_utils.expmap2rotmat(srnn_pred[j, k:k + 3]))


          srnn_pred[:, 0:6] = 0

          # Pick only the dimensions with sufficient standard deviation. Others are ignored.
          idx_to_use = np.where(np.std(srnn_pred, 0) > 1e-4)[0]

          euc_error = np.power(srnn_gts[:, idx_to_use] - srnn_pred[:, idx_to_use], 2)
          euc_error = np.sum(euc_error, 1)
          euc_error = np.sqrt(euc_error)
          mean_errors.append(euc_error)

      # MAE
      mean_errors = np.array(mean_errors)
      mean_mean_errors = np.mean( mean_errors, 0 )

      print("{0: <16} |".format("milliseconds"), end="")
      for ms in [80, 160, 320, 400, 560, 1000]:
          print(" {0:5d} |".format(ms), end="")
      print()
      print("{0: <16} |".format(action), end="")
      for ms in [1, 3, 7, 9, 13, 24]:
          if seq_length_out >= ms + 1:
              print(" {0:.3f} |".format(mean_mean_errors[ms]), end="")
          else:
              print("   n/a |", end="")
      print()

      # MPJPE & PCK
      xyz_gt = np.array(xyz_gt)
      xyz_pred = np.array(xyz_pred)
      # Evaluation using pck score
      pck = np.zeros(seq_length_out)
      for k in range(seq_length_out):
          l2_norm, euclidean, pck[k], pck_allthreshold = distance_3D(
              xyz_pred[:, k, :],
              xyz_gt[:, k, :])
          print("Predict next %d frames, l2_norm: %f, MPJPE: %f, PCK: %f" % (k + 1, l2_norm, euclidean, pck[k]))
          if k == 24: # 1000ms  # 9:400ms
              print("All threshold:")
              print(pck_allthreshold)

  return


def distance_3D(prediction, ground_truth):
    '''
    Input:
    prediction, ground_truth: (N, P) N: batch_size, P: all dims
    visibility: (N, J), J = P / 3, value either 1 or 0
    bounding_box_length: (N,)
    Output:
    l2_norm: l2 norm of difference of the pose vectors
    euclidean: average euclidean distance of each joints
    pck: percentage of prediction joints in threshold distance from ground_truth
    '''
    N, P = prediction.shape
    diff = prediction - ground_truth
    l2_norm = np.average(np.linalg.norm(diff, axis = 1))

    x = diff[:, ::3]
    y = diff[:, 1::3]
    z = diff[:, 2::3]
    joint_distance = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    euclidean = np.average(joint_distance)
    visibility = np.ones((N, int(P/3))) # all joints are visible for h36m

    # pck: ignore invisible joints
    # we normalize (x, y) to -1 ~ 1 before training,
    # so pck with threshold 0.05 counts the joints whose distance < 0.1
    in_threshold_distance = (joint_distance < 20) * visibility
    pck = np.sum(in_threshold_distance) / np.sum(visibility)

    pck_allthreshold = np.zeros(7)
    for i, thres in enumerate([0, 25, 50, 100, 150, 200, 250]):#[0, 5, 10, 15, 20, 25, 30,35,40]
        in_threshold_distance_all = (joint_distance < thres) * visibility
        pck_allthreshold[i] = np.sum(in_threshold_distance_all) / np.sum(visibility)

    return l2_norm, euclidean, pck, pck_allthreshold

def define_actions( action ):
  """
  Define the list of actions we are using.

  Args
    action: String with the passed action. Could be "all"
  Returns
    actions: List of strings of actions
  Raises
    ValueError if the action is not included in H3.6M
  """

  actions = ["walking", "eating", "smoking", "discussion",  "directions",
              "greeting", "phoning", "posing", "purchases", "sitting",
              "sittingdown", "takingphoto", "waiting", "walkingdog",
              "walkingtogether"]

  if action in actions:
    return [action]

  if action == "all":
    return actions

  if action == "all_srnn":
    return ["walking", "eating", "smoking", "discussion"]

  raise( ValueError, "Unrecognized action: %d" % action )


def main(_):
  evaluate()

if __name__ == "__main__":
  tf.app.run()
