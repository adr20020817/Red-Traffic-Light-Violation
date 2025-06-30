import requests
import base64

# Beem API Configuration
API_KEY = "383bcac14e9c302d"
APP_SECRET = "YmE2NzQ5Y2JkNDdhNzQ5YmQwMjc0N2Q2OWZlODdlOGY4NTQ1ODM1YTYwNDE2NGI0NDNhYTUwZGRjMzE4MmNjMg=="

def send_sms(phone_number, message, license_plate=None, area_of_violation="traffic intersection"):

    try:
        # Encode the credentials
        credentials = f"{API_KEY}:{APP_SECRET}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()

        headers = {
            "Authorization": f"Basic {encoded_credentials}",
            "Content-Type": "application/json"
        }

        # Use custom message or create default violation message
        if message is None and license_plate:
            formatted_message = f"Your vehicle with plate number {license_plate} was detected violating the red lights at the {area_of_violation}. If it wasn't you please notify the authorities otherwise you are penalized. Thanks."
        else:
            formatted_message = message

        payload = {
            "source_addr": "BUS POA",  # Your approved sender ID
            "schedule_time": "",
            "encoding": "0",
            "message": formatted_message,
            "recipients": [
                {
                    "recipient_id": 1,
                    "dest_addr": phone_number
                }
            ]
        }

        response = requests.post(
            "https://apisms.beem.africa/v1/send",
            json=payload,
            headers=headers,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ SMS sent successfully to {phone_number}")
            print(f"Response: {result}")
            return True
        else:
            print(f"❌ SMS failed. Status: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ SMS sending error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error sending SMS: {e}")
        return False

def send_violation_sms(phone_number, owner_name, license_plate, area="traffic intersection"):

    message = f"Dear {owner_name}, your vehicle [{license_plate}] violated a red light at {area}. Please respond if this wasn't you. Thanks."
    return send_sms(phone_number, message, license_plate, area)

def test_sms():
    """Test function to send a sample SMS"""
    test_number = "+255711995201"
    test_message = "This is a test message from your traffic violation system."
    return send_sms(test_number, test_message)

if __name__ == "__main__":
    # Test the SMS functionality
    print("Testing SMS functionality...")
    test_sms()