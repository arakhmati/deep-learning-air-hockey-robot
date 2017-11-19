/*Bluetooth Module     MEGA2560
  RXD                TXD1(pin18)
  TXD                RXD1(pin19)
  VCC                5V
  GND                GND
*/ 
boolean toggle = false;
void setup() {
  pinMode(19, INPUT);  
  digitalWrite(19, HIGH);
  
  //define 2 serial port
  Serial.begin(9600);
  Serial1.begin(9600); 

  pinMode(13, OUTPUT);

}

void loop() {
  toggle = !toggle;
  digitalWrite(13,toggle);
  delay(500);
  char c;
  //the IDE send,phone receive
  if (Serial.available()) {
    c = Serial.read();
     Serial1.print(c);
  }
  //the phone send ,IDE receive
  if (Serial1.available()) {
    printf("true1");
    c = Serial1.read();
    Serial.println(c);
  }
}
