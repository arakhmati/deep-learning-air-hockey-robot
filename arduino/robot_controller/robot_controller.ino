#include <Stepper.h>
#include <Wire.h>
#include <EnableInterrupt.h>

//I made a change

/*Bluetooth Module     MEGA2560
  RXD                TXD3(pin14)
  TXD                RXD3(pin15)
  VCC                5V
  GND                GND
*/
//setup decision vector
char buffer[100];
const char * actions[] { "NW", "W", "SW", "N", "Stand", "S", "NE", "E", "SE", "Undefined" };
volatile int digit = 4;

//Stepper Motor x-axis
int enablex = 22;
int directx = 33;

//Stepper Motor y-axis
int enabley = 23;
int directyL = 30;
int directyR = 29;

//both motors
int pulse = 27;

//interrupt pins
int xleft = 2;
int xright = 3;
int yfront = 18;
int yrear = 19;

//reading the buffer from neural network
int interval = 1;
int intervalCounter = 0;

//level of difficulty setup
int motorSpeed = 1000; // setup starting motor speed to be used
int level = 0;
int levelPin = 10;

void setup() {
  Serial.begin(115200);
  Serial.print("Sketch:   ");   Serial.println(__FILE__);
  Serial.print("Uploaded: ");   Serial.println(__DATE__);
  Serial.println(" ");

  //Initializes buffer for bluetooth connection
  long bt_baud = 115200;
  sprintf (buffer, "BTserial started at %ld", bt_baud);
  Serial3.begin(bt_baud);
  Serial.println(buffer);

  //Stepper setup x-axis
  pinMode(enablex, OUTPUT);
  pinMode(directx, OUTPUT);

  //Stepper setup y-axis
  pinMode(enabley, OUTPUT);
  pinMode(directyL, OUTPUT);
  pinMode(directyR, OUTPUT);

  //Pulse for all motors
  pinMode(pulse, OUTPUT);

  //interrupt pins
  pinMode(xleft, INPUT);
  pinMode(xright, INPUT);
  pinMode(yfront, INPUT);
  pinMode(yrear, INPUT);
  pinMode(levelPin, INPUT);


  enableInterrupt(levelPin, levelI, RISING);  //attaches interrupt to pin 10
  enableInterrupt(xleft,  leftI,  RISING);
  enableInterrupt(xright,  rightI,  RISING);
  enableInterrupt(yfront,  frontI,  RISING);
  enableInterrupt(yrear,  rearI,  RISING);
}

int steps_skipped_after_stand = 5;
int stand_counter = 0;
void loop() {
  //motor speed pulsing
  digitalWrite(pulse, HIGH);
  delayMicroseconds(motorSpeed);
  digitalWrite(pulse, LOW);
  delayMicroseconds(motorSpeed);

  //bluetooth connection retrieves value from n ueral network, converters to digit and passes to control function
  //delay(1);
  char c;
  //waits for value to be sent to buffer from the neural network and then reads value
  if (Serial3.available()) {
    c = Serial3.read();
    if (isdigit(c)) {
      int digit = c - 48; // ACSII to digit conversion

      sprintf(buffer, "Predicted Action is %s", actions[digit]);
      Serial.println(buffer);

      if ((digit == 4 || digit == 9) && stand_counter == 0) {
        stand_counter = steps_skipped_after_stand;
      }
      sprintf(buffer, "Stand Counter is %d", stand_counter);
      Serial.println(buffer);

      if (stand_counter == 0) {
        control(digit, 0); //calls control function which uses digit value to choose direction and enable values to be HIGH or LOW for each case
      }

      stand_counter--;
      if (stand_counter < 0)
        stand_counter = 0;
    }
  }
}




//takes in camera input and selects direction of motors
void control(int digit, int interrupt_delay) {
  delay(100);  //delay between limit switch action to delay motor direction switching so the motors do not stall

  if (interrupt_delay) {
    Serial.println("Interrupt Delay!");
    delay(400);
  }

  //NW direction decision
  if (digit == 0) {
    //turn on x motor
    digitalWrite(enablex, HIGH);
    digitalWrite(directx, HIGH);

    //turn on y motors
    digitalWrite(enabley, HIGH);
    digitalWrite(directyL, HIGH);
    digitalWrite(directyR, LOW);
    Serial.println("NW direction");
  }

  //W direction decision
  if (digit == 1) {
    //turn on x motor
    digitalWrite(enablex, HIGH);
    digitalWrite(directx, HIGH);

    //turn off y motors
    digitalWrite(enabley, LOW);
    Serial.println("W direction");
  }

  //SW direction decision
  if (digit == 2) {
    //turn on x motor
    digitalWrite(enablex, HIGH);
    digitalWrite(directx, HIGH);

    //turn on y motors
    digitalWrite(enabley, HIGH);
    digitalWrite(directyL, LOW);
    digitalWrite(directyR, HIGH);
    Serial.println("SW direction");
  }

  //N direction
  if (digit == 3) {
    //turn off x motor
    digitalWrite(enablex, LOW);

    //turn on y motors
    digitalWrite(enabley, HIGH);
    digitalWrite(directyL, HIGH);
    digitalWrite(directyR, LOW);

    Serial.println("N direction");
  }

  //Stay in position
  if (digit == 4) {
    //turn off x motor
    digitalWrite(enablex, LOW);

    //turn off y motors
    digitalWrite(enabley, LOW);

    Serial.println("STAY");
  }

  //S direction
  if (digit == 5) {
    //turn off x motor
    digitalWrite(enablex, LOW);

    //turn on y motors
    digitalWrite(enabley, HIGH);
    digitalWrite(directyL, LOW);
    digitalWrite(directyR, HIGH);

    Serial.println("S direction");
  }

  //NE direction
  if (digit == 6) {
    //turn on x motor
    digitalWrite(enablex, HIGH);
    digitalWrite(directx, LOW);

    //turn on y motors
    digitalWrite(enabley, HIGH);
    digitalWrite(directyL, HIGH);
    digitalWrite(directyR, LOW);

    Serial.println("NE direction");
  }

  //E direction
  if (digit == 7) {
    //turn on x motor
    digitalWrite(enablex, HIGH);
    digitalWrite(directx, LOW);

    //turn off y motors
    digitalWrite(enabley, LOW);

    Serial.println("E direction");
  }

  //SE direction
  if (digit == 8) {
    //turn on x motor
    digitalWrite(enablex, HIGH);
    digitalWrite(directx, LOW);

    //turn on y motors
    digitalWrite(enabley, HIGH);
    digitalWrite(directyL, LOW);
    digitalWrite(directyR, HIGH);

    Serial.println("SE direction");
  }

  //undefined image
  if (digit == 9) {
    //turn off x motor
    digitalWrite(enablex, LOW);

    //turn off y motors
    digitalWrite(enabley, LOW);

    Serial.println("Undefined");
  }

}


//left limit switch interrupt functions
void leftI() {

  //this function debounces the switch by delaying the call to the control function as the limit switch often does not settle on HIGH or LOW instantly.
  static unsigned long lastTime1 = 0;
  unsigned long currentTime1 = millis();
  if (currentTime1 - lastTime1 > 200) {
    Serial.println("left interrupt");
    controlLeft();
  }
  lastTime1 = currentTime1;
}

//reverses the x-axis motor to move the mallet to the right
void controlLeft() {
  delay(100);
  digitalWrite(directx, LOW);
  control(4, 1);
}

//right limit switch interrupt functions
void rightI() {

  //this function debounces the switch by delaying the call to the control function as the limit switch often does not settle on HIGH or LOW instantly.
  static unsigned long lastTime2 = 0;
  unsigned long currentTime2 = millis();
  if (currentTime2 - lastTime2 > 200) {
    Serial.println("right x direction");
    controlRight();
  }
  lastTime2 = currentTime2;
}


//reverses the x-axis motor to move the mallet to the left
void controlRight() {
  delay(100);
  digitalWrite(directx, HIGH);
  control(4, 1);
}

//front limit switch interrupt functions
void frontI() {

  //this function debounces the switch by delaying the call to the control function as the limit switch often does not settle on HIGH or LOW instantly.
  static unsigned long lastTime3 = 0;
  unsigned long currentTime3 = millis();
  if (currentTime3 - lastTime3 > 200) {
    Serial.println("front y direction");
    controlFront();
  }
  lastTime3 = currentTime3;
}


//reverses the x-axis motor to move the mallet to the rear
void controlFront() {
  delay(100);
  digitalWrite(directyL, LOW);
  digitalWrite(directyR, HIGH);
  control(4, 1);
}

//rear limit switch interrupt function
void rearI() {
  //this function debounces the switch by delaying the call to the control function as the limit switch often does not settle on HIGH or LOW instantly.
  static unsigned long lastTime4 = 0;
  unsigned long currentTime4 = millis();
  if (currentTime4 - lastTime4 > 200) {
    Serial.println("rear y direction");
    controlRear();
  }
  lastTime4 = currentTime4;

}


//reverses the x-axis motor to move the mallet to the front
void controlRear() {
  delay(100);
  digitalWrite(directyL, HIGH);
  digitalWrite(directyR, LOW);
  control(4, 1);
}


//motor speed interrupt function
void levelI() {

  //this function debounces the switch by delaying the call to the control function as the limit switch often does not settle on HIGH or LOW instantly.
  static unsigned long lastTime5 = 0;
  unsigned long currentTime5 = millis();
  if (currentTime5 - lastTime5 > 200) {
    Serial.println("speed control");
    speedControl();
  }
  lastTime5 = currentTime5;

}

//changes the speed of the motor by adjusting the time between pulses
void speedControl() {
  static int motorSpeeds[4] = {100, 300, 500, 750}; //motor speed array which includes the delay between pulses
  if (++level > 3) //increases level from 0 to 3 and reverts back to 0
    level = 0;
  motorSpeed = motorSpeeds[level]; //sets motor speed based on level value and corresponding position in array
  Serial.println(motorSpeed);
}



