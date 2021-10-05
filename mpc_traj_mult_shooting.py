#!/usr/bin/env python3

#Multiple shooting continuous trajectory MPC
#Author: Shivam Sood
#Date: 14/08/2021

import casadi as ca
import numpy as np
import time

from numpy.lib.utils import safe_eval

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

"""
MPC constraints and tuning params
"""
T = 0.05
N = 20
rob_diam = 1
wheel_rad = 1

v_max, v_min, w_max, w_min = 0.22, -0.22, 2.84, -2.84

Q_x = 10000
Q_y = 10000
Q_theta = 800
R1 = 100
R2 = 100

x_init = 0
y_init = 0
theta_init = 0
x_target = -2
y_target = 0
theta_target = np,pi/2


class Turtlebot3Mpc(Node):

	def __init__(self):
		"""************************************************************
		** Initialise variables
		************************************************************"""
		super().__init__('turtlebot3_mpc')
		self.odom = Odometry()
		self.last_pose_x = 0.0
		self.last_pose_y = 0.0
		self.last_pose_theta = 0.0
		self.mpc_iter = 0
		self.u0 = ca.DM.zeros((2, 20))
		# have to somehow add x_0 here
		self.X0 = ca.repmat(ca.DM([0, 0, 0]), 1, N+1)

		"""************************************************************
		** Initialise ROS publishers and subscribers
		************************************************************"""
		qos = QoSProfile(depth=10)

		# Initialise publishers
		self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', qos)

		# Initialise subscribers
		self.odom_sub = self.create_subscription(
			Odometry, 'odom', self.odom_callback, qos)

		"""************************************************************
		** Initialise timers
		************************************************************"""
		self.update_timer = self.create_timer(
			0.010, self.update_callback)  # unit: s

		self.get_logger().info("Turtlebot3 mpc control node has been initialised.")

	"""*******************************************************************************
	** Callback functions and relevant functions
	*******************************************************************************"""

	def odom_callback(self, msg):
		self.last_pose_x = msg.pose.pose.position.x
		self.last_pose_y = msg.pose.pose.position.y
		_, _, self.last_pose_theta = self.euler_from_quaternion(
			msg.pose.pose.orientation)

		# self.init_odom_state = True

	def update_callback(self):
		# if self.init_odom_state is True:
		self.mpc()

	def euler_from_quaternion(self, quat):
		"""
		Convert quaternion (w in last place) to euler roll, pitch, yaw.

		quat = [x, y, z, w]
		"""
		x = quat.x
		y = quat.y
		z = quat.z
		w = quat.w

		sinr_cosp = 2 * (w * x + y * z)
		cosr_cosp = 1 - 2 * (x * x + y * y)
		roll = np.arctan2(sinr_cosp, cosr_cosp)

		sinp = 2 * (w * y - z * x)
		pitch = np.arcsin(sinp)

		siny_cosp = 2 * (w * z + x * y)
		cosy_cosp = 1 - 2 * (y * y + z * z)
		yaw = np.arctan2(siny_cosp, cosy_cosp)

		return roll, pitch, yaw

	def mpc(self):
		x, y, theta = ca.SX.sym('x'), ca.SX.sym('y'), ca.SX.sym('theta')
		states = ca.vertcat(x, y, theta)
		n_states = states.numel()

		v, omega = ca.SX.sym('v'), ca.SX.sym('omega')
		controls = ca.vertcat(v, omega)
		n_controls = controls.numel()
		rhs = ca.vertcat(v*ca.cos(theta), v*ca.sin(theta), omega)
		f = ca.Function('f', [states, controls], [rhs])
		X = ca.SX.sym('X', n_states, N+1)
		U = ca.SX.sym('U', n_controls, N)
		P = ca.SX.sym('P', n_states+N*(n_states+n_controls))

		# X[:,0] = P[0:n_states]
		# for k in range(0,N-1):
		#    st,con = X[:,k],U[:,k]
		#    f_value = f(st,con)
		#    st_next = st+(T*f_value)
		#    X[:,k+1] = st_next

		# print(X)

		# ff = ca.Function('ff',[U,P],[X])

		obj = 0
		g = X[:, 0] - P[:n_states]

		Q = ca.diagcat(Q_x, Q_y, Q_theta)
		R = ca.diagcat(R1, R2)
		step_horizon = 0.1
		"""
		Calculating the objective function as well as constraint eqns in variable form
		Predictions using runge kutta method
		"""
		for k in range(0, N):
			st, con = X[:, k], U[:, k]
			obj = obj + (st - P[5*(k+1)-2:5*(k+1)+1]).T @ Q @ (st - P[5*(k+1)-2:5*(k+1)+1]) + (
				con-P[5*(k+1)+1:5*(k+1)+3]).T @ R @ (con-P[5*(k+1)+1:5*(k+1)+3])
			st_next = X[:, k+1]
			k1 = f(st, con)
			k2 = f(st + step_horizon/2*k1, con)
			k3 = f(st + step_horizon/2*k2, con)
			k4 = f(st + step_horizon * k3, con)
			st_next_RK4 = st + (step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
			g = ca.vertcat(g, st_next - st_next_RK4)

		# print(obj)
		"""
		Initializeing the casadi solver and passing contraint values
		"""
		OPT_variables = ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))

		nlp_prob = {
			'f': obj,
			'x': OPT_variables,
			'g': g,
			'p': P
		}

		opts = {
			'ipopt': {
				'max_iter': 2000,
				'print_level': 0,
				'acceptable_tol': 1e-8,
				'acceptable_obj_change_tol': 1e-6
			},
			'print_time': 0
		}

		solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

		lbx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))
		ubx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))

		lbx[0: n_states*(N+1): n_states] = -ca.inf     
		lbx[1: n_states*(N+1): n_states] = -ca.inf     
		lbx[2: n_states*(N+1): n_states] = -ca.inf     

		ubx[0: n_states*(N+1): n_states] = ca.inf     
		ubx[1: n_states*(N+1): n_states] = ca.inf      
		ubx[2: n_states*(N+1): n_states] = ca.inf      

		lbx[n_states*(N+1):] = v_min                  
		ubx[n_states*(N+1):] = v_max                  

		args = {
				'lbg': ca.DM.zeros((n_states*(N+1), 1)),
				'ubg': ca.DM.zeros((n_states*(N+1), 1)),
				'lbx': lbx,
				'ubx': ubx
			}

		x_cur, y_cur, theta_cur = self.last_pose_x, self.last_pose_y, self.last_pose_theta
		x_0 = ca.DM([x_cur, y_cur, theta_cur])
		xf = ca.DM([x_target, y_target, theta_target])

		# u0 = ca.DM.zeros((n_controls, N))
		self.X0 = ca.repmat(x_0, 1, N+1)
		def apply_mpc(u):
			twist = Twist()
			u_app = u[:, 0]
			v, w = float(u_app[0, 0].full()[0, 0]), float(
				u_app[1, 0].full()[0, 0])
			# vel_right = (v+w*rob_diam)/wheel_rad
			# vel_left = (v-w*rob_diam)/wheel_rad

			# x_cur,y_cur,theta_cur = self.last_pose_x,self.last_pose_y,self.last_pose_theta
			# x_0 = ca.DM([x_cur, y_cur, theta_cur])

			u0 = ca.horzcat(u[:, 1:], ca.reshape(u[:, -1], -1, 1))
			# print(u0)

			twist.linear.x = v
			twist.linear.y = 0.0
			twist.angular.z = w
			# print('x_cur= ',x_cur,'w= ',w)
			self.cmd_vel_pub.publish(twist)
			return u0
		print("error = ", ca.norm_2(x_0 - xf))
		n = []
		n= ca.vertcat(n,x_0)
		current_time = self.mpc_iter*T
		
		"""
		Putting in the varaible values and desired trajecotry
		"""
		for k in range(0,N):
			t_predict = current_time+(k)*T
			x_ref,y_ref,theta_ref = 1,0.5*t_predict,0
			u_ref, omega_ref = 0.5,0
			if y_ref>=4:
				x_ref,y_ref,theta_ref = 1,4,0
				u_ref, omega_ref = 0.5,0
			n = ca.vertcat(n,x_ref,y_ref,theta_ref)
			n = ca.vertcat(n,u_ref,omega_ref)
		args['p'] = ca.vertcat(n)
		#print(args['p'])
		args['x0'] = ca.vertcat(ca.reshape(self.X0, n_states*(N+1), 1),ca.reshape(self.u0, n_controls*N, 1))
		
		sol = solver(x0=args['x0'],lbx=args['lbx'],ubx=args['ubx'],lbg=args['lbg'],ubg=args['ubg'],p=args['p'])
		
		u = ca.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
		# u = ca.reshape(sol['x'].T, n_controls, N) 
		X0 = ca.reshape(sol['x'][: n_states * (N+1)], n_states, N+1)
		
		X0 = ca.horzcat(
			X0[:, 1:],
			ca.reshape(X0[:, -1], -1, 1)
		)

		
		self.u0 = apply_mpc(u)#,u0 = apply_mpc(u)
		self.mpc_iter +=1
		

def main(args=None):
	
	rclpy.init(args=args)

	turtlebot3_mpc = Turtlebot3Mpc()

	rclpy.spin(turtlebot3_mpc)

	turtlebot3_mpc.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
