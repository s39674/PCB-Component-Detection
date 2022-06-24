# PCB-Component-Detection

**PCB-CD** is one of the engines that in the future will power **image2schematic**: https://github.com/s39674/Image2schematic. You can also use it as a standalone tool to identify pcb components. Any help is greatly appreciated!

## Testing

### training

For training the model, you will need some dataset to work with. The only one I could find is the pcb_wacv_2019 dataset: https://sites.google.com/view/chiawen-kuo/home/pcb-component-detection which includes a number of PCBs labeled with their components. You can use `extract_pcb_wacv_2019.py` to extract each component to a unique folder. I have uploaded the `.csv` file but you can also get it with `extract_pcb_wacv_2019.py`.

`extract_pcb_wacv_2019.py` assumes you have this folder structure:
(there is a one-liner to create all of this below)
```bash
|-- pcb_wacv_2019_formatted
|   |-- battery
|   |-- button
|   |-- buzzer
|   |-- capacitor
|   |-- clock
|   |-- connector
|   |-- diode
|   |-- display
|   |-- emi_filter
|   |-- ferrite_bead
|   |-- fuse
|   |-- heatsink
|   |-- ic
|   |-- inductor
|   |-- jumper
|   |-- led
|   |-- potentiometer
|   |-- resistor
|   |-- transformer
|   |-- transistor
```
one-liner:
```bash
mkdir pcb_wacv_2019_formatted && cd pcb_wacv_2019_formatted/ && mkdir battery button buzzer capacitor clock connector diode display emi_filter ferrite_bead fuse heatsink ic inductor jumper led potentiometer resistor transformer transistor
```

Now set `imgWriteEnable = True` in `extract_pcb_wacv_2019.py` and run it. You should see all of the components in their respective folder. You could now run `train.py`, setting `training = False`. You should now see `pcbComponent_net.pth`.
