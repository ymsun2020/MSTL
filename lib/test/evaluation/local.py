from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.network_path = './tracking/networks/'    # Where tracking networks are stored.
    settings.save_dir ='.tracking'
    settings.result_plot_path = '/home/suixin/MyPaper/Code/HTCT/HTCT820/lib/test/result_plots/'
    settings.results_path = './lib/test/tracking_results/'    # Where to store tracking results
    settings.uav_path = './UAV123'
    return settings

