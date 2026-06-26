# ROBOTICS Reference — PID Controller Library
# Provides: PIDController, CascadePID, PID2D

import numpy as np

class PIDController:
    """1D PID with anti-windup clamp, output saturation, and reset."""
    def __init__(self, Kp, Ki, Kd, setpoint=0., dt=0.01,
                 out_min=-float("inf"), out_max=float("inf"),
                 anti_windup=True, integral_clamp=100.):
        self.Kp=Kp; self.Ki=Ki; self.Kd=Kd
        self.setpoint=setpoint; self.dt=dt
        self.out_min=out_min; self.out_max=out_max
        self.anti_windup=anti_windup; self.integral_clamp=integral_clamp
        self.integral=0.; self.prev_error=0.

    def step(self, measurement):
        error   = self.setpoint - measurement
        P       = self.Kp * error
        D       = self.Kd * (error - self.prev_error) / self.dt
        self.prev_error = error
        raw     = P + self.Ki * self.integral + D
        output  = np.clip(raw, self.out_min, self.out_max)
        if self.anti_windup and output != raw:
            pass
        else:
            self.integral = np.clip(self.integral + error * self.dt,
                                    -self.integral_clamp, self.integral_clamp)
        return output

    def reset(self):
        self.integral = 0.; self.prev_error = 0.

    def set_setpoint(self, sp):
        self.setpoint = sp


class CascadePID:
    """
    Two-loop cascade controller.
    Outer loop: setpoint → velocity setpoint
    Inner loop: velocity setpoint → control force/torque
    Usage:
        ctrl = CascadePID(outer_params, inner_params, dt)
        force = ctrl.step(pos, vel)
    """
    def __init__(self, outer_kpid, inner_kpid, dt=0.01,
                 pos_setpoint=0., inner_out_min=-50., inner_out_max=50.):
        self.outer = PIDController(*outer_kpid, dt=dt,
                                   out_min=-10., out_max=10.)
        self.inner = PIDController(*inner_kpid, dt=dt,
                                   out_min=inner_out_min, out_max=inner_out_max)
        self.outer.setpoint = pos_setpoint

    def step(self, position, velocity):
        vel_sp = self.outer.step(position)
        self.inner.setpoint = vel_sp
        return self.inner.step(velocity)

    def set_setpoint(self, sp):
        self.outer.setpoint = sp


class PID2D:
    """2D PID for planar position control (e.g. robot arm end-effector)."""
    def __init__(self, Kp=2., Ki=0.1, Kd=0.5, dt=0.1,
                 max_force=float("inf"), integral_clamp=20.):
        self.Kp=Kp; self.Ki=Ki; self.Kd=Kd; self.dt=dt
        self.max_force=max_force; self.integral_clamp=integral_clamp
        self.integral  = np.zeros(2)
        self.prev_error= np.zeros(2)
        self.setpoint  = np.zeros(2)

    def step(self, position):
        err = self.setpoint - np.asarray(position, dtype=float)
        self.integral = np.clip(self.integral + err*self.dt,
                                -self.integral_clamp, self.integral_clamp)
        D   = (err - self.prev_error) / self.dt
        self.prev_error = err
        out = self.Kp*err + self.Ki*self.integral + self.Kd*D
        n   = np.linalg.norm(out)
        return out/n*self.max_force if n>self.max_force else out

    def set_setpoint(self, sp): self.setpoint = np.asarray(sp, dtype=float)
    def reset(self): self.integral[:]=0.; self.prev_error[:]=0.


# ─── Quick demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    pid = PIDController(Kp=8., Ki=5., Kd=3., setpoint=1., dt=0.01,
                        out_min=-50., out_max=50.)
    x=xdot=0.; xs=[]
    for _ in range(600):
        F = pid.step(x)
        xddot = (F - 0.5*xdot - 2.*x)/1.
        xdot += xddot*0.01; x+=xdot*0.01; xs.append(x)

    plt.figure(figsize=(8,3))
    plt.plot(xs); plt.axhline(1.,color="r",linestyle="--",label="Setpoint")
    plt.title("PIDController — mass-spring-damper"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("pid_demo.png",dpi=80); plt.close()
    print(f"1D PID final={xs[-1]:.4f}  Saved pid_demo.png")

    casc = CascadePID((2.,0.5,0.1),(5.,1.,0.2), dt=0.01, pos_setpoint=1.)
    pos=vel=0.; ps=[]
    for _ in range(800):
        F = casc.step(pos, vel)
        vel += F*0.01 - 0.2*vel*0.01; pos += vel*0.01; ps.append(pos)
    print(f"Cascade PID final={ps[-1]:.4f}")
