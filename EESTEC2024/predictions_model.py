import os
import json
import argparse
import tensorflow as tf
from scapy.all import rdpcap
from scapy.layers.inet import IP, TCP

# Căile către modelul salvat și rezultatele predicțiilor
output_model_path = "./content/model1.h5"
output_predictions_path = "./content/predictions.json"


def pcap_to_json(pcap_file):

    # Read packets from the pcap file
    packets = rdpcap(pcap_file)

    # Dictionary to hold reassembled TCP streams by connection (source IP, dest IP, source port, dest port)
    tcp_streams = {}

    for packet in packets:
        if IP in packet and TCP in packet and packet[TCP].payload:
            # Create a unique connection identifier
            connection_id = (packet[IP].src, packet[IP].dst, packet[TCP].sport, packet[TCP].dport)

            # Append the payload to the corresponding stream
            if connection_id not in tcp_streams:
                tcp_streams[connection_id] = b""
            tcp_streams[connection_id] += bytes(packet[TCP].payload)

    # Attempt to parse JSON from reassembled streams
    json_data = []

    for stream_data in tcp_streams.values():
        try:
            # Attempt to decode the stream as JSON
            data = json.loads(stream_data.decode('utf-8'))
            json_data.append(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Skip streams that aren't valid JSON
            pass

    # Display the extracted JSON data
    if json_data:
        for entry in json_data:
            return(json.dumps(entry, indent=2))
    else:
        print("No JSON data found in the pcap file.")



# Funcție pentru a citi și a preprocesa fișierele JSON
def load_data(directory):
    data = []
    filenames = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as f:
            json_string = pcap_to_json(file_path)
            json_data = json.loads(json_string)

            combined_text = " ".join([
                json_data.get("process", {}).get("command_line", ""),
                json_data.get("winlog", {}).get("event_id", ""),
                json_data.get("winlog", {}).get("provider_name", ""),
                json_data.get("winlog", {}).get("task", ""),
                json_data.get("log", {}).get("level", ""),
                json_data.get("event", {}).get("action", ""),
                json_data.get("message", ""),
            ])

            data.append(combined_text)
            filenames.append(filename)

    return data, filenames

def main(test_dir):
    # Încarcă modelul antrenat
    model = tf.keras.models.load_model(output_model_path)
    print(f"Modelul a fost încărcat din: {output_model_path}")

    # Încarcă datele de test
    test_texts, test_filenames = load_data(test_dir)

    # Reconstruim și adaptăm stratul de vectorizare
    vectorizer = tf.keras.layers.TextVectorization(max_tokens=2000, output_mode='int', output_sequence_length=150)
    vectorizer.adapt(test_texts)

    # Transformăm textele de test în secvențe numerice
    X_test = vectorizer(tf.constant(test_texts))

    # Realizăm predicții pe setul de date de test
    predictions = model.predict(X_test)
    predicted_labels = [int(tf.argmax(pred).numpy()) for pred in predictions]

    # Salvează rezultatele în format JSON
    output = {filename: label for filename, label in zip(test_filenames, predicted_labels)}
    with open(output_predictions_path, 'w') as f:
        json.dump(output, f)

    print("Predicțiile au fost salvate în:", output_predictions_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict labels using a trained model.")
    parser.add_argument("test_dir", type=str, help="Calea către directorul de test.")
    args = parser.parse_args()

    main(test_dir=args.test_dir)
