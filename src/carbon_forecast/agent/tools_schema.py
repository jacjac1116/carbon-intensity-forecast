TOOLS = [

        # -----------------------------------
        # FEATURE ↔ ERROR CORRELATION
        # -----------------------------------

        {
            "name": "get_error_correlation",

            "description": (
                "Analyse how a specific feature relates to model prediction errors. "
                "Returns the Pearson correlation between the feature values and absolute prediction error, "
                "along with average model error when the feature is in its lowest and highest quartiles. "
                "Also returns the error ratio and difference between low and high feature regimes. "
                "Use this tool when investigating which features are associated with larger model failures "
                "or whether model performance degrades under certain operating conditions."
            ),

            "input_schema": {

                "type": "object",

                "properties": {

                    "feature_name": {

                        "type": "string",

                        "description": (
                            "Name of the feature column to analyse. "
                            "Examples include weather variables, demand proxies, "
                            "generation mix features, or lagged variables."
                        )
                    }
                },

                "required": ["feature_name"]
            }
        },

        # -----------------------------------
        # FAILURE VS SUCCESS COMPARISON
        # -----------------------------------

        {
            "name": "compare_failure_vs_success",

            "description": (
                "Compare how a feature behaves during model failures versus normal operation. "
                "Failures are defined as predictions whose absolute error exceeds a threshold "
                "relative to overall MAE. "
                "Returns mean and standard deviation of the feature during failures and successes, "
                "plus the differences between them. "
                "Use this tool to identify whether specific conditions are overrepresented during failures, "
                "such as high demand, low wind, or extreme temperatures."
            ),

            "input_schema": {

                "type": "object",

                "properties": {

                    "feature_name": {

                        "type": "string",

                        "description": (
                            "Name of the feature column to compare between "
                            "failure and success periods."
                        )
                    }
                },

                "required": ["feature_name"]
            }
        },

        # -----------------------------------
        # EPISODE DEEP DIVE
        # -----------------------------------

        {
            "name": "get_episode_deep_dive",

            "description": (
                "Perform detailed analysis for a specific time window or operational event. "
                "Returns duration, mean and maximum prediction error, "
                "plus summary statistics for all features during the selected period. "
                "Use this tool to investigate major forecast failures, storms, "
                "demand spikes, unusual weather events, or anomalous behaviour."
            ),

            "input_schema": {

                "type": "object",

                "properties": {

                    "start": {

                        "type": "string",

                        "description": (
                            "Start datetime for the analysis window in ISO format. "
                            "Example: '2024-01-15 00:00:00'"
                        )
                    },

                    "end": {

                        "type": "string",

                        "description": (
                            "End datetime for the analysis window in ISO format. "
                            "Example: '2024-01-16 12:00:00'"
                        )
                    }
                },

                "required": ["start", "end"]
            }
        },

        # -----------------------------------
        # EXTREME FEATURE ANALYSIS
        # -----------------------------------

        {
            "name": "get_feature_at_extremes",

            "description": (
                "Evaluate model performance when a feature reaches unusually high values. "
                "Returns the percentile threshold, the corresponding feature value, "
                "average prediction error under extreme conditions, and the number of affected samples. "
                "Use this tool to determine whether the model struggles during extreme events "
                "such as high carbon intensity, strong wind generation, heatwaves, "
                "or unusually high electricity demand."
            ),

            "input_schema": {

                "type": "object",

                "properties": {

                    "feature_name": {

                        "type": "string",

                        "description": (
                            "Name of the feature column to analyse "
                            "under extreme conditions."
                        )
                    },

                    "percentile": {

                        "type": "number",

                        "description": (
                            "Percentile threshold used to define extreme values. "
                            "For example, 0.95 analyses the top 5% highest values."
                        ),

                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },

                "required": [
                    "feature_name",
                    "percentile"
                ]
            }
        }
    ]