#include "flightlib/objects/quadrotor.hpp"

namespace flightlib {

Quadrotor::Quadrotor(const std::string &cfg_path)
  : world_box_((Matrix<3, 2>() << -100, 100, -100, 100, -100, 100).finished()),
    size_(1.0, 1.0, 1.0),
    collision_(false) {
  //
  YAML::Node cfg = YAML::LoadFile(cfg_path);

  // create quadrotor dynamics and update the parameters
  dynamics_.updateParams(cfg);
  init();
}

Quadrotor::Quadrotor(const QuadrotorDynamics &dynamics)
  : world_box_((Matrix<3, 2>() << -100, 100, -100, 100, -100, 100).finished()),
    dynamics_(dynamics),
    size_(1.0, 1.0, 1.0),
    collision_(false) {
  init();
}

Quadrotor::~Quadrotor() {}

bool Quadrotor::run(const Command &cmd, const Scalar ctl_dt) {
  // std::cout<<"total thrust: "<<cmd.collective_thrust<<std::endl;
  if (!setCommand(cmd)) return false;
  // std::cout<<"IAM TRUEEE"<<std::endl;
  return run(ctl_dt);
}

bool Quadrotor::run(const Scalar ctl_dt) {
  if (!state_.valid()) return false;
  if (!cmd_.valid()) return false;
  // std::cout<<"IAM RUNNINGGGG"<<std::endl;
  QuadState old_state = state_;
  QuadState next_state = state_;

  // time
  const Scalar max_dt = integrator_ptr_->dtMax();
  Scalar remain_ctl_dt = ctl_dt;

  // simulation loop
  while (remain_ctl_dt > 0.0) {
    const Scalar sim_dt = std::min(remain_ctl_dt, max_dt);

    const Vector<4> motor_thrusts_des =
      cmd_.isSingleRotorThrusts() ? cmd_.thrusts
                                  : runFlightCtl(sim_dt, state_.w, cmd_);

    // std::cout<<"START "<<motor_thrusts_des<<" END"<<std::endl;
    // std::cout<<"Motorstrta"<<moto
    runMotors(sim_dt, motor_thrusts_des);
    // motor_thrusts_ = cmd_.thrusts;

    // std::cout<<"ALLOCATION: "<<B_allocation_<<" B END"<<std::endl;
    // std::cout<<"INV ALLOCATION: "<<B_allocation_inv_<<" B INV END"<<std::endl;

    Vector<4> force_torques = B_allocation_ * motor_thrusts_;   

    // Compute linear acceleration and body torque
    const Vector<3> force(0.0, 0.0, force_torques[0]);
    state_.a = state_.q() * force * 1.0 / dynamics_.getMass() + gz_;
    // auto vel = state_.v;
    // Vector<3> motor_axis(0, 0, 1)
    // auto v_parallel = np.dot(vel, motor_axis) * motor_axis;
    // vel = vel - v_parallel;
   

    // f_drag = -omega * C_DRAG * v_perp;
    // m_drag_induced = np.cross(r, f_drag)

    // force_torques[3] = force_torques

    // compute body torque
    // std::cout<<"START: "<<force_torques.segment<3>(1)<<"END "<<std::endl;
    state_.tau = force_torques.segment<3>(1);

  

    // dynamics integration
    integrator_ptr_->step(state_.x, sim_dt, next_state.x);  

    // update state and sim time
    state_.qx /= state_.qx.norm();

    //
    state_.x = next_state.x;
    remain_ctl_dt -= sim_dt;
  }
  state_.t += ctl_dt;
  //
  constrainInWorldBox(old_state);
  return true;
}

void Quadrotor::init(void) {
  // reset
  updateDynamics(dynamics_);
  reset();
}

bool Quadrotor::reset(void) {
  state_.setZero();
  motor_omega_.setZero();
  motor_thrusts_.setZero();
  return true;
}

bool Quadrotor::reset(const QuadState &state) {
  if (!state.valid()) return false;
  state_ = state;
  motor_omega_.setZero();
  motor_thrusts_.setZero();
  return true;
}

Vector<4> Quadrotor::runFlightCtl(const Scalar sim_dt, const Vector<3> &omega,
                                  const Command &command) {
  // std::cout<<"collective: "<<command.collective_thrust<<std::endl;
  const Scalar force = dynamics_.getMass() * command.collective_thrust;
  // std::cout<<"I AM HERE";
  const Vector<3> omega_err = command.omega - omega;

  Vector<3> body_torque_des =
    dynamics_.getJ() * Kinv_ang_vel_tau_ * omega_err +
    state_.w.cross(dynamics_.getJ() * state_.w);
  // body_torque_des[2] = 0.0;

  // const Vector<3> body_torque_des = dynamics_.getJ()*command.omega + omega.cross(dynamics_.getJ()*omega);

  const Vector<4> thrust_and_torque(force, body_torque_des.x(),
                                    body_torque_des.y(), body_torque_des.z());

  // std::cout<<"START Thrust and torque : "<<thrust_and_torque<<"End thrust and torque"<<std::endl; 
  // Eigen::MatrixXd M(4,4);
  // Eigen::Matrix<Scalar, 4, 4> M;
  // float L, K;

  // L = 0.5;
  // K = 5;
  // M << 1.0/4, -1.0/(4*L), 1.0/(4*L), -1.0/(4*K),
  //         1.0/4, 1.0/(4*L), -1.0/(4*L), -1.0/(4*K),
  //         1.0/4, 1.0/(4*L), 1.0/(4*L), 1.0/(4*K),
  //         1.0/4, -1.0/(4*L), -1.0/(4*L), 1.0/(4*K);

  const Vector<4> motor_thrusts_des = B_allocation_inv_ * thrust_and_torque;

  // std::cout<<"allocated: : "<<motor_thrusts_des<<"End allocated"<<std::endl;
  motor_history_ = motor_thrusts_des;

  // std::cout<<B_allocation_inv_<<std::endl;

  return dynamics_.clampThrust(motor_thrusts_des);
  // std::cout<<"START Motor: "<<motor_thrusts_des<<"END motors"<<std::endl;
}

void Quadrotor::runMotors(const Scalar sim_dt,
                          const Vector<4> &motor_thruts_des) {
  // std::cout<<"DESIRED START: "<<motor_thruts_des<<"END DESIRED"<<std::endl;
  // const Vector<4> motor_omega_des =
  //   dynamics_.motorThrustToOmega(motor_thruts_des);
  // const Vector<4> motor_omega_clamped =
  //   dynamics_.clampMotorOmega(motor_omega_des);

  const Vector<4> motor_omega_des = (motor_thruts_des / dynamics_.getMaxThrust()).array().sqrt().matrix() * dynamics_.motor_omega_max_;
  const Vector<4> motor_omega_clamped =
    dynamics_.clampMotorOmega(motor_omega_des);

  // std::cout<<"OMEGA_BEF STRAT: "<<motor_omega_clamped<<"OMEGABEF END"<<std::endl;

  // simulate motors as a first-order system
  // const Scalar c = std::exp(-sim_dt * dynamics_.getMotorTauInv());
  // motor_omega_ = c * motor_omega_ + (1.0 - c) * motor_omega_clamped;
  // if(motor_omega_clamped > motor_omega_)
  // {
  //   const Scalar c = std::exp(-sim_dt * dynamics_.motor_tau_up_inv_);
  //   motor_omega_ = c * motor_omega_ + (1.0 - c) * motor_omega_clamped;
  // }
  // else{
  //   const Scalar c = std::exp(-sim_dt * dynamics_.motor_tau_down_inv_);
  //   motor_omega_ = c * motor_omega_ + (1.0 - c) * motor_omega_clamped;
  // }
  for (int i = 0; i < 4; ++i) {
    const bool spin_up = motor_omega_clamped(i) > motor_omega_(i);

    const Scalar tau_inv = spin_up
        ? dynamics_.motor_tau_up_inv_
        : dynamics_.motor_tau_down_inv_;

    const Scalar c = std::exp(-sim_dt * tau_inv);
    // std::cout<<tau_inv<<std::endl;

    motor_omega_(i) =
        c * motor_omega_(i) + (1.0 - c) * motor_omega_clamped(i);

  }
  // std::cout<<"OMEGA_AF STRAT: "<<motor_omega_<<"OMEGA_AF END"<<std::endl;
  
  motor_thrusts_ = 8.54858e-6 * motor_omega_.array().square().matrix();
  
  // std::cout<<"SENT START: "<<motor_thrusts_<<"SENT DESIRED"<<std::endl;
  // motor_thrusts_ = dynamics_.motorOmegaToThrust(motor_omega_);
  motor_thrusts_ = dynamics_.clampThrust(motor_thrusts_);

  // std::cout<<"SENT START2: "<<motor_thrusts_<<"SENT DESIRED2"<<std::endl;
}

bool Quadrotor::setCommand(const Command &cmd) {
  // std::cout<<"i am commadning"<<std::endl;
  if (!cmd.valid()) return false;
  // std::cout<<"IAM VALID"<<std::endl;
  cmd_ = cmd;

  if (std::isfinite(cmd_.collective_thrust)){
    // std::cout<<"YOYOFINITE"<<std::endl;
    cmd_.collective_thrust = dynamics_.clampThrust(cmd_.collective_thrust);
  }

  if (cmd_.omega.allFinite()) cmd_.omega = dynamics_.clampBodyrates(cmd_.omega);

  if (cmd_.thrusts.allFinite())
    cmd_.thrusts = dynamics_.clampThrust(cmd_.thrusts);

  return true;
}

bool Quadrotor::setState(const QuadState &state) {
  if (!state.valid()) return false;
  state_ = state;
  return true;
}

bool Quadrotor::setWorldBox(const Ref<Matrix<3, 2>> box) {
  if (box(0, 0) >= box(0, 1) || box(1, 0) >= box(1, 1) ||
      box(2, 0) >= box(2, 1)) {
    return false;
  }
  world_box_ = box;
  return true;
}

Vector<4> Quadrotor::getHistory()
{
  return motor_history_;
}


bool Quadrotor::constrainInWorldBox(const QuadState &old_state) {
  if (!old_state.valid()) return false;

  // violate world box constraint in the x-axis
  if (state_.x(QS::POSX) < world_box_(0, 0) ||
      state_.x(QS::POSX) > world_box_(0, 1)) {
    state_.x(QS::POSX) = old_state.x(QS::POSX);
    state_.x(QS::VELX) = 0.0;
  }

  // violate world box constraint in the y-axis
  if (state_.x(QS::POSY) < world_box_(1, 0) ||
      state_.x(QS::POSY) > world_box_(1, 1)) {
    state_.x(QS::POSY) = old_state.x(QS::POSY);
    state_.x(QS::VELY) = 0.0;
  }

  // violate world box constraint in the x-axis
  if (state_.x(QS::POSZ) <= world_box_(2, 0) ||
      state_.x(QS::POSZ) > world_box_(2, 1)) {
    //
    state_.x(QS::POSZ) = world_box_(2, 0);

    // reset velocity to zero
    state_.x(QS::VELX) = 0.0;
    state_.x(QS::VELY) = 0.0;

    // reset acceleration to zero
    state_.a << 0.0, 0.0, 0.0;
    // reset angular velocity to zero
    state_.w << 0.0, 0.0, 0.0;
  }
  return true;
}

bool Quadrotor::getState(QuadState *const state) const {
  if (!state_.valid()) return false;

  *state = state_;
  return true;
}

bool Quadrotor::getMotorThrusts(Ref<Vector<4>> motor_thrusts) const {
  motor_thrusts = motor_thrusts_;
  return true;
}

bool Quadrotor::getMotorOmega(Ref<Vector<4>> motor_omega) const {
  motor_omega = motor_omega_;
  return true;
}

bool Quadrotor::getDynamics(QuadrotorDynamics *const dynamics) const {
  if (!dynamics_.valid()) return false;
  *dynamics = dynamics_;
  return true;
}

const QuadrotorDynamics &Quadrotor::getDynamics() { return dynamics_; }

bool Quadrotor::updateDynamics(const QuadrotorDynamics &dynamics) {
  if (!dynamics.valid()) {
    // std::cout << "[Quadrotor] dynamics is not valid!" << std::endl;
    return false;
  }
  dynamics_ = dynamics;
  integrator_ptr_ =
    std::make_unique<IntegratorRK4>(dynamics_.getDynamicsFunction(), 2.5e-3);

  B_allocation_ = dynamics_.getAllocationMatrix();
  B_allocation_inv_ = B_allocation_.inverse();
  return true;
}

bool Quadrotor::addRGBCamera(std::shared_ptr<RGBCamera> camera) {
  rgb_cameras_.push_back(camera);
  return true;
}

Vector<3> Quadrotor::getSize(void) const { return size_; }

Vector<3> Quadrotor::getPosition(void) const { return state_.p; }

std::vector<std::shared_ptr<RGBCamera>> Quadrotor::getCameras(void) const {
  return rgb_cameras_;
};

bool Quadrotor::getCamera(const size_t cam_id,
                          std::shared_ptr<RGBCamera> camera) const {
  if (cam_id <= rgb_cameras_.size()) {
    return false;
  }

  camera = rgb_cameras_[cam_id];
  return true;
}

void Quadrotor::massrandomization()
{
  // static std::random_device rd;            // non-deterministic seed
  static std::mt19937 gen(0);            // Mersenne Twister RNG
  static std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);

  dynamics_.setMass(dist(gen));

}

bool Quadrotor::getCollision() const { return collision_; }

}  // namespace flightlib
