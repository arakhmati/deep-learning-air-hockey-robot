/*Bluetooth Module     MEGA2560
  RXD                TXD1(pin18)
  TXD                RXD1(kpin19)
  VCC                5V
  GND                GND
*/

#define DELAY 1
#define LED_PIN 13

const char *actions[] {
        "NW", "W", "SW", "N", "Stand", "S", "NE", "E", "SE"
};

boolean toggle = false;

void setup()
{
  pinMode(LED_PIN, OUTPUT);

  Serial.begin(115200);
  Serial.print("Sketch:   ");   Serial.println(__FILE__);
  Serial.print("Uploaded: ");   Serial.println(__DATE__);
  Serial.println(" ");

  Serial1.begin(115200);
  Serial.println("BTserial started at 9600");

}

void loop()
{

    delay(1);
    char c;
    
    if (Serial1.available()) {
        c = Serial1.read();
        if (isdigit(c)) {
            int digit = c - 48; // ACSII to digit
            if (digit < 9) {
                Serial.write(actions[digit]);
            }
            else
                Serial.write(c);
        }
        else {
            Serial.write(c);
        }
        
        toggle = !toggle;
        digitalWrite(LED_PIN, toggle);
    }
}
