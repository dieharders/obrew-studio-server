# How to Connect External Apps to Obrew Server

There are two ways app developers can connect their web or native apps to the Obrew backend server:

## 1. Internal (simplest)

When starting Obrew, choose `Connection Options`:

- Click the app card from the list of supported app clients or manually enter the address of your app, and click `Start`.

## 2. External

If you are developing a native app or your webapp is hosted online, you must enable SSL to enable https:

- From the Obrew Server start menu, choose `Settings`
- Under `Server Options`, check `Enable SSL` and be sure to refer to [README](deploy.md) to setup certificates.

* Note, if your webapp or native app is in-development and/or being served locally over http, then you won't need to enable SSL on the server, but you may still need to add the app address to `Whitelists` in `Settings`.

[Back to README](../README.md)
