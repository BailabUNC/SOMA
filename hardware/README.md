# SOMA hardware

| Folder | Contents |
|---|---|
| `firmware/TRAiLL_DAQ_Pico/` | RP2040 C source, CMake project and Pico SDK import helper |
| `firmware/LED_driver/` | Arduino LED-driver sketch |
| `cad/` | Existing DWG asset and Altium/AutoCAD/Blender desktop helpers |

Firmware and CAD content is preserved without behavioral changes. Configure a
local Raspberry Pi Pico SDK/toolchain and use
`hardware/firmware/TRAiLL_DAQ_Pico` as the CMake source directory for the Pico
project. No firmware was built or flashed during Python reorganization.

CAD Python helpers operate the foreground desktop through mouse/keyboard
automation. Install `python -m pip install -e '.[cad]'` and run them individually
in the intended desktop application. They are not imported by the training
package. See [acquisition](../src/traill_daq/README.md) for the Python serial app.
