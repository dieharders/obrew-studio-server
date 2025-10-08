# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this folder.

## Purpose

Render a GUI before the main Python process for Server takes over.

Files in the `/public` folder render a frontend UI in vanilla html/js/css. It is intended as a light-weight GUI that offers settings for starting the server and a WebUI app.

## Function

It passes data directly to and from the main Python process.
