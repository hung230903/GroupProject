import pywifi
import time
import numpy as np
import joblib
from flask import Flask, jsonify, render_template
import threading

app = Flask(__name__)

# Load the saved Random Forest models and encoder
rf_model_x = joblib.load('rf_model_x.pkl')
rf_model_y = joblib.load('rf_model_y.pkl')
encoder = joblib.load('encoder.pkl')

# Global variable to store the current coordinates
current_coordinates = {'x': 0, 'y': 0}


# Function to scan for available Wi-Fi networks
def scan_wifi():
    wifi = pywifi.PyWiFi()  # Initialize the PyWiFi object
    iface = wifi.interfaces()[0]  # Get the first wireless interface
    iface.scan()  # Start scanning for Wi-Fi networks

    print("\nScanning for networks...")
    time.sleep(3)  # Wait for scan results

    networks = iface.scan_results()  # Get scan results
    print(f"Scan complete. Found {len(networks)} networks.")

    # Print list of available networks with SSID and index
    print(f"\n{'Index':<5}{'SSID':<30}{'Signal (dBm)':<15}")
    print("=" * 50)
    for idx, network in enumerate(networks):
        ssid = network.ssid
        signal = network.signal
        print(f"{idx:<5}{ssid:<30}{signal:<15}")

    return networks  # Return the scanned networks


# Function to automatically select routers with SSIDs starting with USTH or ICT
def choose_routers(networks):
    selected_routers = [network for network in networks if network.ssid.startswith(('USTH', 'ICT'))]

    if not selected_routers:
        print("\nNo networks found with SSIDs starting with 'USTH' or 'ICT'. Please check your environment.")
    else:
        print(f"\nAutomatically selected {len(selected_routers)} routers with matching SSIDs:")
        for router in selected_routers:
            print(f"SSID: {router.ssid}, Signal: {router.signal}")

    return selected_routers


# Function to predict coordinates based on signal strengths
def predict_coordinates(routers, networks):
    # Create SSID_MAC for the selected routers
    ssid_mac_list = []
    router_signals = []

    for router in routers:
        ssid_mac = f"{router.ssid}_{router.bssid}"  # Combine SSID and MAC address
        ssid_mac_list.append(ssid_mac)

        # Find the corresponding signal strength for the router
        signal_strength = next(
            (network.signal for network in networks if network.ssid == router.ssid and network.bssid == router.bssid),
            0,  # Default to 0 if the network is not found
        )
        router_signals.append(signal_strength)

    # Encode SSID_MAC using the encoder
    encoded_ssid_mac = encoder.transform(np.array(ssid_mac_list).reshape(-1, 1))

    # Combine encoded SSID_MAC with signal strengths for prediction
    input_data = np.hstack((encoded_ssid_mac, np.array(router_signals).reshape(-1, 1)))

    # Predict the X and Y coordinates using the trained Random Forest models
    x_pred = rf_model_x.predict(input_data)
    y_pred = rf_model_y.predict(input_data)

    return x_pred[0], y_pred[0]


# Update monitor_and_predict to handle the corrected prediction process
def monitor_and_predict(routers):
    wifi = pywifi.PyWiFi()
    iface = wifi.interfaces()[0]

    try:
        while True:
            iface.scan()  # Continuously scan for updated signal strengths
            time.sleep(3)  # Sleep for 3 seconds to let the scan complete
            networks = iface.scan_results()

            # Predict coordinates based on the router signals and update current_coordinates
            predicted_coordinates = predict_coordinates(routers, networks)
            current_coordinates['x'] = predicted_coordinates[0]
            current_coordinates['y'] = predicted_coordinates[1]
            print(f'Predicted Coordinates: x={predicted_coordinates[0]}, y={predicted_coordinates[1]}\n')

            print("Monitoring... Press Ctrl+C to stop.")
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/coordinates')
def coordinates():
    return jsonify({'x': float(current_coordinates['x']), 'y': float(current_coordinates['y'])})


# Main Program Execution
if __name__ == "__main__":
    # Step 1: Scan available networks
    available_networks = scan_wifi()

    # Step 2: Automatically select routers with specific SSIDs
    selected_routers = choose_routers(available_networks)

    if selected_routers:
        # Step 3: Start the scanning thread
        scan_thread = threading.Thread(target=monitor_and_predict, args=(selected_routers,))
        scan_thread.daemon = True  # Allow the thread to be killed when the main program exits
        scan_thread.start()

        # Run the Flask app
        app.run()
    else:
        print("\nNo suitable routers were selected. Exiting...")
