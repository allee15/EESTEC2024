import os
import json
import argparse
import tensorflow as tf
from scapy.all import rdpcap
from scapy.layers.inet import IP, TCP

# Paths to save the model and vectorizer layer
output_model_path = "./content/model13.h5"
vectorizer_model_path = "./content/vectorizer_layer"  # Directory for auxiliary model


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
            return json.dumps(entry, indent=2)
    else:
        print("No JSON data found in the pcap file.")


def json_to_string(data):
    def extract_values(obj):
        values = []
        if isinstance(obj, dict):
            # If the object is a dictionary, recurse on each value
            for value in obj.values():
                values.extend(extract_values(value))
        elif isinstance(obj, list):
            # If the object is a list, recurse on each item
            for item in obj:
                values.extend(extract_values(item))
        elif isinstance(obj, (str, int, float)):
            # If the object is a string, int, or float, add it to the values list
            values.append(str(obj))
        return values

    # Join all extracted values into a single string separated by spaces
    return " ".join(extract_values(data))


# Function to load and preprocess JSON files
def load_data(directory):
    data = []
    labels = []

    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist.")
        return data, labels

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as f:
            json_string = pcap_to_json(file_path)
            json_data = json.loads(json_string)

            # Convert JSON data to a single string of text
            combined_text = json_to_string(json_data)

            data.append(combined_text)
            labels.append(json_data.get("label", 0))  # Default label can be 0 if not present

    return data, labels


def main(train_dir):
    # Load training data
    train_texts, train_labels = load_data(train_dir)

    # Create a text vectorization layer with increased settings for diverse words and longer sequences
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=5000,  # Increased vocabulary size to account for diverse words
        output_mode='int',
        output_sequence_length=200  # Increased sequence length to capture more context
    )
    vectorizer.adapt(train_texts)

    # Transform texts into numeric sequences
    X_train = vectorizer(tf.constant(train_texts))

    # Convert labels to the appropriate format
    y_train = tf.keras.utils.to_categorical(train_labels, 2)

    # Define the main model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(200,)),
        tf.keras.layers.Dense(512, activation='tanh'),
        tf.keras.layers.Dense(256, activation='tanh'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    # Compile and train the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    class_weight = {0: 1.0, 1: 5.0}
    model.fit(X_train, y_train, epochs=50, batch_size=4, verbose=1, class_weight=class_weight)

    # Save the main model
    model.save(output_model_path)
    print(f"Trained model saved to: {output_model_path}")

    # Save the vectorizer model in SavedModel format
    vectorizer_model = tf.keras.Sequential([vectorizer])
    tf.saved_model.save(vectorizer_model, vectorizer_model_path)
    print(f"Vectorizer saved to: {vectorizer_model_path}")


if __name__ == '__main__':
    print(tf.__version__)
    parser = argparse.ArgumentParser(description="Train a model on the specified directory.")
    parser.add_argument("train_dir", type=str, help="Path to the training directory.")
    args = parser.parse_args()

    main(train_dir=args.train_dir)
