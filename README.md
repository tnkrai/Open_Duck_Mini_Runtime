# Open Duck Mini Runtime

## Raspberry Pi zero 2W setup

### Install Raspberry Pi OS

Download Raspberry Pi OS Lite (64-bit) from here : https://www.raspberrypi.com/software/operating-systems/

Follow the instructions here to install the OS on the SD card : https://www.raspberrypi.com/documentation/computers/getting-started.html

With the Raspberry Pi Imager, you can pre-configure session, wifi and ssh. Do it like below :

![imager_setup](https://github.com/user-attachments/assets/7a4987b2-de83-41dd-ab7f-585259685f16)

> Tip: I configure the rasp to connect to my phone's hotspot, this way I can connect to it from anywhere.

### Setup SSH (If not setup during the installation)

When first booting on the rasp, you will need to connect a screen and a keyboard. The first thing you should do is connect to a wifi network and enable SSH.

To do so, you can follow this guide : https://www.raspberrypi.com/documentation/computers/configuration.html#setting-up-wifi

Then, you can connect to your rasp using SSH without having to plug a screen and a keyboard.

### Update the system and install necessary stuff

```bash
sudo apt update
sudo apt upgrade
sudo apt install git
sudo apt install python3-pip
sudo apt install python3-virtualenvwrapper
(optional) sudo apt install python3-picamzero

```

Add this to the end of the `.bashrc`:

```bash
export WORKON_HOME=$HOME/.virtualenvs
export PROJECT_HOME=$HOME/Devel
source /usr/share/virtualenvwrapper/virtualenvwrapper.sh
```

### Enable I2C

`sudo raspi-config` -> `Interface Options` -> `I2C`

TODO set 400KHz ?

### Set the usbserial latency timer

```bash
cd  /etc/udev/rules.d/
sudo touch 99-usb-serial.rules
sudo nano 99-usb-serial.rules
# copy the following line in the file
SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"
```

### Set the udev rules for the motor control board

TODO


### Setup xbox one controller over bluetooth

Turn your xbox one controller on and set it in pairing mode by long pressing the sync button on the top of the controller.

Run the following commands on the rasp :

```bash
bluetoothctl
scan on
```

Wait for the controller to appear in the list, then run :

```bash
pair <controller_mac_address>
trust <controller_mac_address>
connect <controller_mac_address>
```

The led on the controller should stop blinking and stay on.

You can test that it's working by running

```bash
python3 mini_bdx_runtime/mini_bdx_runtime/xbox_controller.py
```

## Speaker wiring and configuration
Follow this tutorial

> For now, don't activate `/dev/zero` when they ask

https://learn.adafruit.com/adafruit-max98357-i2s-class-d-mono-amp?view=all


## Install the runtime

### Make a virtual environment and activate it

```bash
mkvirtualenv -p python3 open-duck-mini-runtime
workon open-duck-mini-runtime
```

Clone this repository on your rasp, cd into the repo, then :

```bash
git clone https://github.com/apirrone/Open_Duck_Mini_Runtime
cd Open_Duck_Mini_Runtime
git checkout v2
pip install -e .
```

In Raspberry Pi 5, you need to perform the following operations

```bash
pip uninstall -y RPi.GPIO
pip install lgpio
```


## Test the IMU

```bash
python3 mini_bdx_runtime/mini_bdx_runtime/raw_imu.py
```

You can also run `python3 scripts/imu_server.py` on the robot and `python3 scripts/imu_client.py --ip <robot_ip>` on your computer to check that the frame is oriented correctly. 

> To find the ip address of the robot, run `ifconfig` on the robot

## Test motors

This will allow you to verify all your motors are connected and configured.

```bash
python3 scripts/check_motors.py
```

## Make your duck_config.json

Copy `example_config.json` in the home directory of your duck and rename it `duck_config.json`.

`cp example_config.json ~/duck_config.json`

In this file, you can configure some stuff, like registering if you installed the expression features, installed the imu upside down or and other stuff. You also write the joints offsets of your duck here

## Find the joints offsets

This script will guide you through finding the joints offsets of your robot that you can then write in your `duck_config.json`

> This procedure won't be necessary in the future as we will be flashing the offsets directly in each motor's eeprom.

```bash
cd scripts/
python find_soft_offsets.py
```

## Run the walk !

Download the [latest policy checkpoint ](https://github.com/apirrone/Open_Duck_Mini/blob/v2/BEST_WALK_ONNX_2.onnx) and copy it to your duck.

`cd scripts/`

`python v2_rl_walk_mujoco.py --onnx_model_path <path_to>/BEST_WALK_ONNX_2.onnx`



```
- The commands are : 
- A to pause/unpause
- X to turn on/off the projector
- B to play a random sound
- Y to turn on/off head control (very experimental, I don't recommend trying that, it can break your duck's head)
- left and right triggers to control the left and right antennas
- LB (new!) press and hold to increase the walking frequency, kind of a sprint mode 🙂
```

## Installed policies (`~/.tnkr/policies`)

The duck can hold a few policies besides the one this repo ships, and Tnkr Studio installs
them over the robot's HTTP API (`POST /api/policy/install`). They live here:

```
~/.tnkr/policies/
├── active                  # which policy the next walk starts on. Absent = the built-in
└── <policy-id>/
    ├── model.onnx
    ├── manifest.json       # shapes read off the graph + measured inference latency
    └── last_used           # empty; its mtime decides what gets evicted first
```

Facts worth knowing before you go looking:

- **The built-in policy is not in there.** It is resolved through the same `scripts/*.onnx`
  glob the walk has always used, so it tracks whatever the repo ships and cannot be
  evicted. A duck that has never installed anything has an empty (or absent) store and
  walks exactly as it always did.
- **The store is bounded**: at most three installed policies plus the built-in, least
  recently used evicted to make room, and an install is refused outright if it would leave
  less than 200 MB free. A full SD card is a Pi that will not boot, which is a worse
  problem than a policy that walks badly.
- **Getting back is one call**: `curl -X POST http://<duck>:8000/api/policy/select -d
  '{"id":"builtin"}'`. It works with an empty store, a corrupt `active` file, and while a
  walk is running (it takes effect on the next start).
- **Deleting the whole directory is safe.** The next walk resolves the built-in.

Anything not resolved through that glob is treated as a custom policy and runs inside the
safety envelope (velocity and joint-limit clamps, tilt and control-budget aborts) whether or
not anyone asked for it.

## Telemetry

The robot sends anonymous usage data so we can find and fix the setup steps and
failures that affect everyone.

**What it sends:** which setup step ran and whether it worked (with the error
text from the setup log when it did not), API request outcomes from the robot
server (endpoint, status, duration, failure cause), how a walk session ended
(duration, exit code, and whether joint data was streaming, as a true/false),
IMU calibration failures, and hardware facts: Pi model, RAM, OS, Python version,
and which servo USB adapter chip you have (CH343 or FTDI).

**What it never sends:** joint or motion data, your recordings, your wifi
password, session tokens, Supabase credentials, your Tnkr password, your name,
your hostname or username, or your location. Location is off at the source:
every event sets `$geoip_disable`, so PostHog never derives a place from the IP
the request arrived on. The robot is identified only by a random UUID made on
first install.

### If you sign in to Tnkr Studio and connect this robot

We link this robot's anonymous id to your Tnkr account. Everything the robot did
before you signed in, including the whole setup, then shows up as yours rather
than as an anonymous robot.

That is a real change to what "anonymous" means here, so it is worth being exact
about when it happens:

- It happens only if you made a Tnkr account, signed in to Studio, and connected
  this robot. Nothing on the robot initiates it, and it cannot happen otherwise.
- It happens once. The first account to connect a robot is the one it stays
  linked to.
- Turning telemetry off on the robot prevents it completely.

The robot cannot record an owner and does not know one. `GET
/api/telemetry/identity` returns this robot's anonymous UUID and the on/off flag,
and nothing else: no account, no owner, no name. It is read-only, because this
server authenticates nobody and could only ever believe whoever asked first.
Ownership is decided by Tnkr's backend, which does verify who is asking. With
telemetry off, that endpoint reports `{"enabled": false}` and withholds the id,
so opting out here also stops the robot from being linked to any account.

### Turning it off

Either of these works, any time:

- Set `"enabled": false` in `~/.tnkr-telemetry.json`. This is the durable one: it
  covers the robot server and the setup script, and it survives reinstalls,
  including `setup.sh --clean`.
- Set `TNKR_TELEMETRY=0` in the environment, for example by uncommenting the
  `Environment=` line in `/etc/systemd/system/tnkr-robot.service`. Re-running
  `setup.sh` rewrites that service file, so prefer the JSON file if you want it
  to stick.

The setup script shows a notice before it sends anything, and asks first when you
run it interactively.

### Selling or giving away your duck

Reflash the SD card. That gives the robot a new anonymous id and unlinks it from
your account, so its history stops being attached to you and the next owner
starts clean. It also wipes your wifi password and your SSH keys, which you want
to do anyway.

Reflashing is the only way to do this. `setup.sh --clean` deliberately keeps
`~/.tnkr-telemetry.json`, because that file is where your opt-out lives and a
reinstall must not quietly turn telemetry back on.
