import json
import time
import random
import logging
from datetime import datetime
from kafka import KafkaProducer
from faker import Faker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Kafka
KAFKA_BOOTSTRAP_SERVERS = 'kafka1:29092'
KAFKA_TOPIC = 'loan-applications'

fake = Faker()

def create_producer():
    """Create a Kafka producer"""
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
        logger.info(f"Connected to Kafka at {KAFKA_BOOTSTRAP_SERVERS}")
        return producer
    except Exception as e:
        logger.error(f"Failed to connect to Kafka: {e}")
        raise

def generate_loan_data():
    """Generate a sample loan application record"""
    # Generate base data
    person_income = random.randint(12000, 72000)
    home_ownership = random.choice(['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
    loan_amount = random.randint(1000, 35000)
    loan_intent = random.choice(['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
    loan_int_rate = round(random.uniform(10.0, 20.0), 1)
    loan_percent_income = round(loan_amount / person_income, 2)
    previous_defaults = random.choice(['Yes', 'No'])
    
    # Create data record
    data = {
        'timestamp': datetime.now().isoformat(),
        'application_id': fake.uuid4(),
        'person_income': person_income,
        'person_home_ownership': home_ownership,
        'loan_amnt': loan_amount,
        'loan_intent': loan_intent,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'previous_loan_defaults_on_file': previous_defaults
    }
    
    # Determine label based on a simple heuristic
    risk_score = 0
    
    # Risk factors
    if previous_defaults == 'Yes':
        risk_score += 50
    if loan_percent_income > 0.4:
        risk_score += 25
    if loan_int_rate > 15:
        risk_score += 15
    if person_income < 30000:
        risk_score += 20
    if home_ownership == 'RENT':
        risk_score += 10
    if loan_intent in ['PERSONAL', 'MEDICAL']:
        risk_score += 5
        
    # Add label (0 = no default, 1 = default)
    data['label'] = 1 if risk_score > 60 or random.random() < 0.15 else 0
    
    return data

def main():
    """Main function to generate and send loan data to Kafka"""
    # Wait for Kafka to be ready
    time.sleep(5)  # Reduce initial wait time
    
    producer = create_producer()
    
    try:
        count = 0
        while True:
            data = generate_loan_data()
            logger.debug(f"Generated data: {data}")  # Debugging line for tracking data generation
            
            key = data['application_id']
            
            # Send data to Kafka immediately
            producer.send(KAFKA_TOPIC, key=key, value=data)
            print(data)
            count += 1
            
            if count % 100 == 0:
                producer.flush()  # Flush after every 100 records to avoid delay
                logger.info(f"Sent {count} loan applications to Kafka")
                
                # Log sample of last data sent
                if count % 500 == 0:
                    logger.info(f"Sample data: {json.dumps(data, indent=2)}")
            
            # Random delay to simulate realistic data flow (but much faster than before)
            time.sleep(0.1)  # Reduced delay to simulate faster streaming
            
    except KeyboardInterrupt:
        logger.info("Producer interrupted")
    except Exception as e:
        logger.error(f"Error in producer: {e}")
    finally:
        if producer:
            producer.flush()
            producer.close()
            logger.info("Producer closed")

if __name__ == "__main__":
    main()
