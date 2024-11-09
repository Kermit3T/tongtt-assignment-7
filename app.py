from flask import Flask, render_template, request, url_for, session, redirect, make_response, flash
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
from datetime import timedelta
from flask_session import Session
import logging

app = Flask(__name__)
app.secret_key = "your_secret_key_here"
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
app.config['SESSION_FILE_DIR'] = 'flask_session'  # Create this directory in your project
Session(app)

@app.before_request
def before_request():
    session.permanent = True

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_response(template, **kwargs):
    response = make_response(render_template(template, **kwargs))
    # Update CSP headers to be more permissive for development
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-src 'self'"
    )
    return response

def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate initial dataset
    X = np.random.uniform(0, 1, N)
    error = np.random.normal(0, np.sqrt(sigma2), N)
    Y = beta0 + beta1 * X + mu + error

    # Fit linear regression model
    model = LinearRegression()
    X_reshaped = X.reshape(-1, 1)
    model.fit(X_reshaped, Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Generate scatter plot with regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, alpha=0.5)
    plt.plot(X, model.predict(X_reshaped), color='red', label='Fitted Line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot with Regression Line')
    plt.legend()
    plt.grid(True)
    plt.savefig('static/plot1.png')
    plt.close()

    # Run simulations
    slopes = np.zeros(S)
    intercepts = np.zeros(S)
    
    for i in range(S):
        X_sim = np.random.uniform(0, 1, N)
        error_sim = np.random.normal(0, np.sqrt(sigma2), N)
        Y_sim = beta0 + beta1 * X_sim + mu + error_sim
        
        sim_model = LinearRegression()
        X_sim_reshaped = X_sim.reshape(-1, 1)
        sim_model.fit(X_sim_reshaped, Y_sim)
        slopes[i] = sim_model.coef_[0]
        intercepts[i] = sim_model.intercept_

    # Plot histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.hist(slopes, bins=30, alpha=0.7, color='blue')
    ax1.axvline(slope, color='red', linestyle='dashed', label='Observed')
    ax1.axvline(beta1, color='green', linestyle='dashed', label='True')
    ax1.set_title('Distribution of Slopes')
    ax1.legend()
    
    ax2.hist(intercepts, bins=30, alpha=0.7, color='blue')
    ax2.axvline(intercept, color='red', linestyle='dashed', label='Observed')
    ax2.axvline(beta0, color='green', linestyle='dashed', label='True')
    ax2.set_title('Distribution of Intercepts')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('static/plot2.png')
    plt.close()

    # Calculate proportions of extreme values
    slope_more_extreme = np.mean(np.abs(slopes - beta1) >= np.abs(slope - beta1))
    intercept_extreme = np.mean(np.abs(intercepts - beta0) >= np.abs(intercept - beta0))

    # Convert numpy arrays to lists for JSON serialization in session
    slopes_list = slopes.tolist()
    intercepts_list = intercepts.tolist()

    return (X, Y, slope, intercept, 'static/plot1.png', 'static/plot2.png',
            slope_more_extreme, intercept_extreme, slopes_list, intercepts_list)

def calculate_p_value(simulated_stats, observed_stat, hypothesized_value, test_type):
    """
    Calculate p-value for hypothesis test
    """
    simulated_stats = np.array(simulated_stats)  # Ensure numpy array
    
    if test_type == "greater":
        # H₀: parameter ≤ hypothesized_value vs H₁: parameter > hypothesized_value
        p_value = np.mean(simulated_stats >= observed_stat)
    elif test_type == "less":
        # H₀: parameter ≥ hypothesized_value vs H₁: parameter < hypothesized_value
        p_value = np.mean(simulated_stats <= observed_stat)
    else:  # "not_equal"
        # H₀: parameter = hypothesized_value vs H₁: parameter ≠ hypothesized_value
        differences = np.abs(simulated_stats - hypothesized_value)
        observed_difference = np.abs(observed_stat - hypothesized_value)
        p_value = np.mean(differences >= observed_difference)
    
    return float(p_value)  # Convert to Python float for JSON serialization

@app.route("/", methods=["GET", "POST"])
def index():
    print("Index route accessed")
    print("Current session data:", dict(session))
    print("Request method:", request.method)
    
    if request.method == "POST":
        print("Processing POST request")
        try:
            # Get form data
            form_data = {
                "N": int(request.form["N"]),
                "mu": float(request.form["mu"]),
                "sigma2": float(request.form["sigma2"]),
                "beta0": float(request.form["beta0"]),
                "beta1": float(request.form["beta1"]),
                "S": int(request.form["S"])
            }
            print("Form data received:", form_data)

            # Generate data
            (X, Y, slope, intercept, plot1, plot2, slope_extreme, intercept_extreme,
             slopes, intercepts) = generate_data(**form_data)

            # Update session data
            session_data = {
                "X": X.tolist(),
                "Y": Y.tolist(),
                "slope": float(slope),
                "intercept": float(intercept),
                "slopes": slopes,
                "intercepts": intercepts,
                "slope_extreme": float(slope_extreme),
                "intercept_extreme": float(intercept_extreme),
                **form_data,
                "data_generated": True
            }
            
            # Update session
            for key, value in session_data.items():
                session[key] = value
            
            print("Updated session data:", dict(session))

            return create_response(
                "index.html",
                plot1=plot1,
                plot2=plot2,
                slope_extreme=slope_extreme,
                intercept_extreme=intercept_extreme,
                **form_data,
                data_generated=True
            )
        except Exception as e:
            print(f"Error in index POST: {str(e)}")
            return create_response("index.html", error=str(e))

    # For GET requests
    if session.get("data_generated"):
            print("Returning existing session data")
            return create_response(
                "index.html",
                plot1="static/plot1.png",
                plot2="static/plot2.png",
                slope_extreme=session.get("slope_extreme"),
                intercept_extreme=session.get("intercept_extreme"),
                N=session.get("N"),
                mu=session.get("mu"),
                sigma2=session.get("sigma2"),
                beta0=session.get("beta0"),
                beta1=session.get("beta1"),
                S=session.get("S"),
                data_generated=True
            )
        
    print("No session data, returning empty form")
    return create_response(
        "index.html", 
        data_generated=False,
        plot1=None,
        plot2=None,
        plot3=None,
        plot4=None
    )

@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    print("Current session data:", dict(session))
    print("Form data received:", dict(request.form))
    
    if not session.get("data_generated"):
        print("data_generated flag not found in session")
        flash("Please generate data first", "warning")
        return redirect(url_for("index"))
    
    try:
        # Get stored values from session
        N = session.get("N")
        S = session.get("S")
        slope = session.get("slope")
        intercept = session.get("intercept")
        slopes = session.get("slopes")
        intercepts = session.get("intercepts")
        beta0 = session.get("beta0")
        beta1 = session.get("beta1")

        # Validate that all required data exists
        # Changed validation to check each value individually
        required_data = {
            'N': N, 'S': S, 'slope': slope, 'intercept': intercept,
            'slopes': slopes, 'intercepts': intercepts, 'beta0': beta0, 'beta1': beta1
        }
        
        missing = [key for key, value in required_data.items() if value is None]
        if missing:
            flash(f"Missing required data: {', '.join(missing)}", "danger")
            return redirect(url_for("index"))

        # Get form data
        parameter = request.form.get("parameter")
        test_type = request.form.get("test_type")

        if not parameter or not test_type:
            flash("Please select both a parameter and test type", "warning")
            return redirect(url_for("index"))

        # Convert types
        N = int(N)
        S = int(S)
        slope = float(slope)
        intercept = float(intercept)
        slopes = np.array(slopes)
        intercepts = np.array(intercepts)
        beta0 = float(beta0)
        beta1 = float(beta1)

        if parameter == "slope":
            simulated_stats = slopes
            observed_stat = slope
            hypothesized_value = beta1
        else:
            simulated_stats = intercepts
            observed_stat = intercept
            hypothesized_value = beta0

        p_value = calculate_p_value(simulated_stats, observed_stat, hypothesized_value, test_type)
        logger.info(f"Calculated p-value: {p_value}")
        
        fun_message = None
        if p_value <= 0.0001:
            fun_message = "Wow! You've found an extremely rare event! 🎯"

        # Plot histogram with test results
        plt.figure(figsize=(10, 6))
        plt.hist(simulated_stats, bins=30, alpha=0.7, density=True)
        plt.axvline(observed_stat, color='red', linestyle='dashed', label='Observed')
        plt.axvline(hypothesized_value, color='green', linestyle='dashed', label='Hypothesized')
        plt.title(f'Distribution under H₀ (p-value = {p_value:.4f})')
        plt.legend()
        plt.grid(True)
        plt.savefig('static/plot3.png')
        plt.close()

        return create_response(
            "index.html",
            plot1="static/plot1.png",
            plot2="static/plot2.png",
            plot3="static/plot3.png",
            parameter=parameter,
            observed_stat=observed_stat,
            hypothesized_value=hypothesized_value,
            p_value=p_value,
            fun_message=fun_message,
            N=N, beta0=beta0, beta1=beta1, S=S,
            data_generated=True,
            slope_extreme=session.get("slope_extreme"),
            intercept_extreme=session.get("intercept_extreme")
        )
    except Exception as e:
        flash(f"Error: {str(e)}", "danger")
        logger.error(f"Error in hypothesis_test: {str(e)}", exc_info=True)
        return redirect(url_for("index"))
    
@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Check if data has been generated
    if not session.get("data_generated"):
        return redirect(url_for("index"))
    
    try:
        # Get stored values from session
        N = session.get("N")
        mu = session.get("mu")
        sigma2 = session.get("sigma2")
        beta0 = session.get("beta0")
        beta1 = session.get("beta1")
        S = session.get("S")
        slope = session.get("slope")
        intercept = session.get("intercept")
        slopes = session.get("slopes")
        intercepts = session.get("intercepts")

        # Validate that all required data exists
        if None in [N, mu, sigma2, beta0, beta1, S, slope, intercept, slopes, intercepts]:
            return redirect(url_for("index"))

        # Convert types
        N = int(N)
        mu = float(mu)
        sigma2 = float(sigma2)
        beta0 = float(beta0)
        beta1 = float(beta1)
        S = int(S)
        slope = float(slope)
        intercept = float(intercept)
        slopes = np.array(slopes)
        intercepts = np.array(intercepts)

        parameter = request.form.get("parameter")
        confidence_level = request.form.get("confidence_level")

        if not parameter or not confidence_level:
            return redirect(url_for("index"))

        confidence_level = float(confidence_level)

        if parameter == "slope":
            estimates = slopes
            observed_stat = slope
            true_param = beta1
        else:
            estimates = intercepts
            observed_stat = intercept
            true_param = beta0

        # Calculate confidence interval
        mean_estimate = np.mean(estimates)
        std_estimate = np.std(estimates, ddof=1)
        
        alpha = 1 - confidence_level/100
        t_value = stats.t.ppf(1 - alpha/2, S-1)
        margin_error = t_value * (std_estimate / np.sqrt(S))
        
        ci_lower = mean_estimate - margin_error
        ci_upper = mean_estimate + margin_error
        includes_true = ci_lower <= true_param <= ci_upper

        # Create confidence interval plot
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(estimates)), estimates, alpha=0.2, color='gray', label='Simulated')
        plt.axhline(y=mean_estimate, color='blue' if includes_true else 'red', 
                    label='Mean Estimate')
        plt.axhline(y=ci_lower, color='green', linestyle='--', label='CI Bounds')
        plt.axhline(y=ci_upper, color='green', linestyle='--')
        plt.axhline(y=true_param, color='black', label='True Value')
        plt.title(f'{confidence_level}% Confidence Interval for {parameter.capitalize()}')
        plt.legend()
        plt.grid(True)
        plt.savefig('static/plot4.png')
        plt.close()

        return create_response(
            "index.html",
            plot1="static/plot1.png",
            plot2="static/plot2.png",
            plot4="static/plot4.png",
            parameter=parameter,
            confidence_level=confidence_level,
            mean_estimate=mean_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            includes_true=includes_true,
            observed_stat=observed_stat,
            N=N, mu=mu, sigma2=sigma2,
            beta0=beta0, beta1=beta1, S=S,
            data_generated=True,
            slope_extreme=session.get("slope_extreme"),
            intercept_extreme=session.get("intercept_extreme")
        )
    except Exception as e:
        print(f"Error in confidence_interval: {str(e)}")
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)