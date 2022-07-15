# PCB-Component-Detection

**PCB-CD** is one of the engines that in the future will power [**image2schematic**](https://github.com/s39674/Image2schematic). You can also use it as a standalone tool to identify pcb components. Any help is greatly appreciated!

## Testing

* **PCB-CD** requires `numpy`, `torch`, `matplotlib` and `pandas`.
* **PCB-CD** uses a simple label map that can be found in `train.py`.

### training

For training the model, you will need some dataset to work with. The only one I could find is the `pcb_wacv_2019` dataset: https://sites.google.com/view/chiawen-kuo/home/pcb-component-detection which includes a number of PCBs labeled with their components. You can use `extract_pcb_wacv_2019.py` to extract each component to a unique folder. I have uploaded the `.csv` file but you can also get it with `extract_pcb_wacv_2019.py`.

`extract_pcb_wacv_2019.py` assumes you have this folder structure:
(there is a one-liner to create all of this below)
```bash
├── pcb_wacv_2019 # the downloaded zip file
│   ├── ACM-109_Bottom
│   ├── ACM-109_Top
│   ├── ArduinoMega_Bottom
│   ├── ArduinoMega_Top
│   ...
│   ...
├── pcb_wacv_2019_formatted
│   ├── battery
│   ├── button
│   ├── buzzer
│   ├── capacitor
│   ├── clock
│   ├── connector
│   ├── diode
│   ├── display
│   ├── emi_filter
│   ├── ferrite_bead
│   ├── fuse
│   ├── heatsink
│   ├── ic
│   ├── inductor
│   ├── jumper
│   ├── led
│   ├── potentiometer
│   ├── resistor
│   ├── transformer
│   └── transistor
```
one-liner:
```bash
mkdir pcb_wacv_2019_formatted && cd pcb_wacv_2019_formatted/ && mkdir battery button buzzer capacitor clock connector diode display emi_filter ferrite_bead fuse heatsink ic inductor jumper led potentiometer resistor transformer transistor
```

Now set `imgWriteEnable = True` in `extract_pcb_wacv_2019.py` and run it. You should see all of the components in their respective folder. You could now run `train.py`, setting `training = True`. You should now see `pcbComponent_net.pth` Neural Network model that can be used to predict new samples with `predict.py`, just set `img_path` to your image.


## Best result so far: (Please zoom in to see detection label)

![image](https://user-images.githubusercontent.com/98311750/179185040-eec9c563-ca4d-4eda-b527-0c0dfdd6d37c.png)

As you can see, there are a lot of improvement to be done. Any suggestion and help will be greatly appreciated!