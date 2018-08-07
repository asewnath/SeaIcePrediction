This folder contains scripts used to experiment with Random Forests and MLPs, but the code
can be slightly modified to use many different scikit-learn functions



    -data_collection_thick: prepares a dataset comprised of features and their respective
                            ground truth using ice concentration and PIOMAS ice thickness data
    
    -data_collection: prepares a dataset comprised of features and their respective ground
                      truth using ice concentration data
    
    -detrend_vs_raw_experiment: trains models and plots coefficients of determination using
                                detrended and raw data ground truth
    
    -model_testing: prints out test results to an excel spreadsheet using models that have 
                    been trained and saved
    
    -model_tune_plot: tunes models and plots coefficient of determination scores
    
    -model_tune_save: tunes and saves models
    
    -no_detrend_model_tuning: I'm not sure what this is but maybe you'll find useful
                              snippets of code here
    
    -region_seg_experimentation: experimenting with using region mask data and plotting 
                                 the results
    
    -seasonal_model_forecast: trying out seasonal forecast ensemble model (bad idea plz ignore)
    
    -seasonal_model_testing: testing seasonal forecast models (bad idea plz ignore)
        