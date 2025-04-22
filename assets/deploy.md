# Deploy to Production

## Deploy to public hosted internet

For production deployments you will either want to run the server behind a reverse proxy using something like Traefic-Hub (free and opens your self hosted server to public internet using encrypted https protocol).

## Start server on local/cloud network over https

If you wish to deploy this on your private network for local access from any device on that network, you will need to run the server using https which requires SSL certificates. Be sure to set the .env var `ENABLE_SSL`.

Rename the included `.env.example` file to `.env` in the `/_deps` folder and modify the vars accordingly.

This command will create a self-signed key and cert files in your current dir that are good for 100 years. These files should go in the `_deps/public` folder. You should generate your own and overwrite the files in `_deps/public`.

```bash
openssl req -x509 -newkey rsa:4096 -nodes -out public/cert.pem -keyout public/key.pem -days 36500
# OR (an alias for same command as above)
pnpm makecert
```

This should be enough for any webapp served over https to access the server. If you see "Warning: Potential Security Risk Ahead" in your browser when using the webapp, you can ignore it by clicking `advanced` then `Accept the Risk` button to continue.

[Back to README](../README.md)
