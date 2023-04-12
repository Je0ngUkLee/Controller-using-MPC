#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
import numpy  as np

import signal
import sys
import rospy

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg      import Path, Odometry
from mav_msgs.msg      import Actuators

from tf.transformations import euler_from_quaternion
from multiprocessing    import Queue

'''
Ctrl+C : program exit
'''

def signal_handler(signal, frame):
  print('\nYou pressed Ctrl+C!')
  sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


def shift(x_n, u):  
  x_n = np.delete(x_n, 0, axis = 1)
  col_add_list_x = x_n[:, -1]
  x_n = np.c_[x_n, col_add_list_x]
  
  u = np.delete(u, 0, axis = 1)
  col_add_list_u = u[:, -1]
  u = np.c_[u, col_add_list_u]
  return x_n, u



'''
MPC and ROS class
'''

class MPC():
  def __init__(self, n_obs):
    
    self.T = 0.2 #[s]
    self.N = 30
    
    self.n_obstacles = n_obs
    
    self.x      = ca.SX.sym('x')
    self.y      = ca.SX.sym('y')
    self.theta  = ca.SX.sym('theta')
    self.states = ca.vertcat(
      self.x,
      self.y,
      self.theta
    )
    self.n_states = self.states.numel()
    
    self.v        = ca.SX.sym('v')
    self.omega    = ca.SX.sym('omega')
    self.controls = ca.vertcat(
      self.v,
      self.omega
    )
    self.n_controls = self.controls.numel()
    
    self.rhs = ca.vertcat(
      self.v * ca.cos(self.theta),
      self.v * ca.sin(self.theta),
      self.omega
    )
    
    self.Q_x     = 0.1
    self.Q_y     = 0.1
    self.Q_theta = 0.005
    
    self.R1      = 0.005
    self.R2      = 0.005
    
    self.f = ca.Function('f', [self.states, self.controls], [self.rhs])
    self.U = ca.SX.sym('U', self.n_controls, self.N)
    self.X = ca.SX.sym('X', self.n_states, (self.N + 1))
    self.P = ca.SX.sym('P', self.n_states + self.N * (self.n_states + self.n_controls))
    self.Q = ca.diagcat(self.Q_x, self.Q_y, self.Q_theta)
    self.R = ca.diagcat(self.R1, self.R2)
    
    ## lbg, ubg
    self.lbg = ca.DM.zeros((self.n_states * (self.N + 1) + self.n_obstacles * (self.N + 1), 1))
    self.ubg = ca.DM.zeros((self.n_states * (self.N + 1) + self.n_obstacles * (self.N + 1), 1))
    
    
    self.x_max     = ca.inf; self.x_min     = -self.x_max
    self.y_max     = ca.inf; self.y_min     = -self.y_max
    self.theta_max = ca.inf; self.theta_min = -self.theta_max
    
    self.v_max     = 1.0;       self.v_min     = 0.0
    self.omega_max = ca.pi / 3; self.omega_min = -self.omega_max
    
    ## lbx, ubx
    
    self.lbx = ca.DM.zeros((self.n_states * (self.N + 1) + self.n_controls * self.N), 1)
    self.ubx = ca.DM.zeros((self.n_states * (self.N + 1) + self.n_controls * self.N), 1)
    
    self.lbx[0 : self.n_states * (self.N + 1) : self.n_states] = self.x_min
    self.ubx[0 : self.n_states * (self.N + 1) : self.n_states] = self.x_max
    self.lbx[1 : self.n_states * (self.N + 1) : self.n_states] = self.y_min
    self.ubx[1 : self.n_states * (self.N + 1) : self.n_states] = self.y_max
    self.lbx[2 : self.n_states * (self.N + 1) : self.n_states] = self.omega_min
    self.ubx[2 : self.n_states * (self.N + 1) : self.n_states] = self.omega_max
    
    self.lbx[self.n_states * (self.N + 1) : self.n_states * (self.N + 1) + self.n_controls * self.N : self.n_controls]     = self.v_min
    self.ubx[self.n_states * (self.N + 1) : self.n_states * (self.N + 1) + self.n_controls * self.N : self.n_controls]     = self.v_max
    self.lbx[self.n_states * (self.N + 1) + 1 : self.n_states * (self.N + 1) + self.n_controls * self.N : self.n_controls] = self.omega_min
    self.ubx[self.n_states * (self.N + 1) + 1 : self.n_states * (self.N + 1) + self.n_controls * self.N : self.n_controls] = self.omega_max
    
    self.args = {
      'lbg' : self.lbg,
      'ubg' : self.ubg,
      'lbx' : self.lbx,
      'ubx' : self.ubx
    }
    
    self.state_init = ca.DM([0.0, 0.0, 0.0])
    
    self.u0 = ca.DM.zeros(self.N, self.n_controls)
    self.X0 = ca.repmat(self.state_init, 1, self.N + 1)
    
    '''
    ROS setting
    '''
    rospy.init_node('Rbt1_MPC_node')
    
    self.odom_sub  = rospy.Subscriber('/rbt1/odom_ground_truth', Odometry, self.odom_callback1, queue_size = 1)
    self.ref_sub   = rospy.Subscriber('/rbt1/reference_trajectory', Path, self.reference_trajectory_callback, queue_size = 1)
    
    self.cmd_pub = rospy.Publisher('/rbt1/jackal_velocity_controller/cmd_vel', Twist, queue_size = 1)
    self.pre_pub = rospy.Publisher('/rbt1/predict_path', Path, queue_size = 1)
    self.ref_pub = rospy.Publisher('/rbt1/ref_path', Path, queue_size = 1)
    
    self.reference_trajectory = Path()
    self.ref_queue = Queue()  
    self.gazebo_pos = [0.0, 0.0]
    self.gazebo_yaw = 0.0
    
  def odom_callback1(self, msg):
    self.gazebo_pos = [msg.pose.pose.position.x, msg.pose.pose.position.y]
    
    q_list = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
    (roll, pitch, yaw) = euler_from_quaternion(q_list)
    
    self.gazebo_yaw = yaw
    
  def reference_trajectory_callback(self, msg):
    self.reference_trajectory = msg
    self.ref_queue.put(msg)
    
  def cmd_vel_publish(self):
    pub_msg = Twist()
    
    pub_msg.linear.x  = self.u0[:, 0][0]
    pub_msg.angular.z = self.u0[:, 0][1]
    
    self.cmd_pub.publish(pub_msg)
    
  def predictive_traj_publish(self):
    pub_msg = Path()
    
    pub_msg.header.frame_id = 'map'
    pub_msg.header.stamp    = rospy.Time.now()
    
    for i in range(self.N + 1):
      pose_stamped = PoseStamped()
      pose_stamped.pose.position.x = self.X0[:, i][0]
      pose_stamped.pose.position.y = self.X0[:, i][1]
      pub_msg.poses.append(pose_stamped)
      
    self.pre_pub.publish(pub_msg)
    
  def reference_traj_publish(self):
    pub_msg = Path()
    
    pub_msg.header.frame_id = 'map'
    pub_msg.header.stamp    = rospy.Time.now()
    
    for i in range(self.N):
      pose_stamped = PoseStamped()
      pose_stamped.pose.position.x = self.ref[i][0]
      pose_stamped.pose.position.y = self.ref[i][1]
      pub_msg.poses.append(pose_stamped)
      
    self.ref_pub.publish(pub_msg)
    
    
  '''
  function of MPC
  '''
  def while_loop_condition(self):
    return not rospy.is_shutdown()
  
  def state_update(self):
    self.state_init[0 : 2] = self.gazebo_pos
    self.state_init[2]     = self.gazebo_yaw
    
  def setting_solver(self):
    
    self.obj = 0
    self.g   = []
    self.g = self.X[:, 0] - self.P[ : self.n_states]
    
    for k in range(self.N):
      self.st  = self.X[:, k]
      self.con = self.U[:, k]
      
      # if k == self.N - 1:
      #   self.Q_x     = 0.4
      #   self.Q_y     = 0.4
      #   self.Q_theta = 0.005
        
      #   self.Q = ca.diagcat(self.Q_x, self.Q_y, self.Q_theta)
      
      state_error_   = self.st  - self.P[(self.n_states + self.n_controls) * k + 3 : (self.n_states + self.n_controls) * k + 6]
      control_error_ = self.con - self.P[(self.n_states + self.n_controls) * k + 6 : (self.n_states + self.n_controls) * k + 8]
      
      self.obj = self.obj + ca.mtimes([state_error_.T, self.Q, state_error_]) \
                          + ca.mtimes([control_error_.T, self.R, control_error_])
                          
      self.st_next = self.X[:, k + 1]
      self.f_value = self.f(self.st, self.con)
      self.st_next_euler = self.st + (self.T * self.f_value)
      self.g = ca.vertcat(self.g, self.st_next - self.st_next_euler)
      
    if self.n_obstacles != 0:
      self.collision_avoidance()
      
    self.OPT_variables = ca.vertcat(
      self.X.reshape((-1, 1)),
      self.U.reshape((-1, 1))
    )
    
    self.nlp_prob = {
      'f' : self.obj,
      'x' : self.OPT_variables,
      'g' : self.g,
      'p' : self.P
    }
    
    '''
    IPOPT (Interior Point Optimizer) is an open source software package for large-scale nonlinear optimization.
    It can be used to solve general nonlinear programming problems (NLPs)
    '''
    
    self.opts = {
      'ipopt' : {
        'max_iter' : 2000,
        'print_level' : 0,
        'acceptable_tol' : 1e-8,
        'acceptable_obj_change_tol' : 1e-6
      },
      'print_time' : 0
    }
    
    self.solver = ca.nlpsol('solver', 'ipopt', self.nlp_prob, self.opts)
    
  def make_args_p(self):
    self.args['p'] = ca.DM.zeros((self.n_states + self.n_controls) * self.N + self.n_states)
    self.args['p'][0 : 3] = self.state_init
    
  def desired_trajectory(self, cnt):
    self.ref = []
    
    for i in range(self.N):
      # self.x_ref     = cnt + (0.1 / self.N) * i
      self.x_ref     = cnt + i * 0.1
      self.y_ref     = 1.8
      
      self.ref.append([self.x_ref, self.y_ref])
      
    return self.ref
    
  def setting_reference(self):
    for k in range(self.N):
      self.theta_ref = 0.0
      self.u_ref     = 0.5
      self.omega_ref = 0.0
      
      self.args['p'][(self.n_states + self.n_controls) * k + 3] = self.ref[k][0]
      self.args['p'][(self.n_states + self.n_controls) * k + 4] = self.ref[k][1]
      self.args['p'][(self.n_states + self.n_controls) * k + 5] = self.theta_ref
      self.args['p'][(self.n_states + self.n_controls) * k + 6] = self.u_ref
      self.args['p'][(self.n_states + self.n_controls) * k + 7] = self.omega_ref
      
  def reshape_and_init_opt_variable(self):
    self.args['x0'] = ca.vertcat(
      ca.reshape(self.X0, self.n_states * (self.N + 1), 1),
      ca.reshape(self.u0, self.n_controls * self.N, 1)
    )
    
  def call_solver(self):
    self.sol = self.solver(
      x0 = self.args['x0'],
      lbx = self.args['lbx'],
      ubx = self.args['ubx'],
      lbg = self.args['lbg'],
      ubg = self.args['ubg'],
      p = self.args['p']
    )
    
  def get_result(self):
    self.X0 = ca.reshape( self.sol['x'][ : self.n_states * (self.N + 1)], self.n_states, self.N + 1 )
    self.u0 = ca.reshape( self.sol['x'][self.n_states * (self.N + 1) : ], self.n_controls, self.N )
    
  def shift_(self):
    self.X0, self.u0 = shift(self.X0, self.u0)
    
  def collision_avoidance(self):
    
    self.rob_diam = 0.6
    self.safe_dis = 0.5
    self.obs      = [ [4.0, 1.5, 1.0], [10.0, 1.7, 1.0] ]
    
    self.lbg = ca.DM.zeros((self.n_states * (self.N + 1) + self.n_obstacles * (self.N + 1), 1))
    self.ubg = ca.DM.zeros((self.n_states * (self.N + 1) + self.n_obstacles * (self.N + 1), 1))
    self.lbg[self.n_states * (self.N + 1) : ] = -ca.inf   # obstacle (inequality constraint)
    
    for k in range(self.N + 1):
      self.g = ca.vertcat(self.g, -ca.sqrt( (self.X[0, k] - self.obs[0][0])**2 + (self.X[1, k] - self.obs[0][1])**2) + (self.rob_diam + self.obs[0][2])/2 + self.safe_dis)
      self.g = ca.vertcat(self.g, -ca.sqrt( (self.X[0, k] - self.obs[1][0])**2 + (self.X[1, k] - self.obs[1][1])**2) + (self.rob_diam + self.obs[1][2])/2 + self.safe_dis)

    self.args = {
      'lbg' : self.lbg,
      'ubg' : self.ubg,
      'lbx' : self.lbx,
      'ubx' : self.ubx
    }
  
    
def main():
  mpc = MPC(n_obs = 2)
  
  loop_cnt = 0
  t = 0
  r = rospy.Rate(10)
  
  while (mpc.while_loop_condition()):
    
    mpc.state_update()
    mpc.setting_solver()
    mpc.make_args_p()
    mpc.desired_trajectory(t)
    mpc.setting_reference()
    mpc.reshape_and_init_opt_variable()
    mpc.call_solver()
    mpc.get_result()
    mpc.cmd_vel_publish()
    mpc.predictive_traj_publish()
    mpc.reference_traj_publish()
    
    print('-------------------------')
    print('loop_count: {}'.format(loop_cnt))
    print('cmd_vel: {}'.format(mpc.u0[:, 0]))
    print('pos_error: {} m'.format(ca.sqrt( (mpc.state_init[0] - mpc.ref[0][0])**2 + (mpc.state_init[1] - mpc.ref[0][1])**2 )))
    print('\n')
    
    mpc.shift_()
    
    t += 0.05
    loop_cnt += 1
    
    r.sleep()
    
if __name__ == '__main__':
  main()