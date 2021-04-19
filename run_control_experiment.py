from visual_mpc.sim.run_control_experiment import ControlManager

if __name__ == '__main__':
    c = ControlManager(save_dir_prefix='classifier_control')
    c.run()




