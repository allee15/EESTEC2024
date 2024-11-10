from scapy.all import rdpcap
from scapy.layers.inet import IP, TCP
import json

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

