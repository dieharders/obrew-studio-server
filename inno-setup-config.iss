; Script generated by the Inno Setup Script Wizard.
; SEE THE DOCUMENTATION FOR DETAILS ON CREATING INNO SETUP SCRIPT FILES!

#define MyAppName "Obrew Studio"
#define MyAppVersion "0.10.0"
#define MyAppPublisher "OpenBrewAi"
#define MyAppURL "https://www.openbrewai.com/"
#define MyAppExeName "Obrew-Studio.exe"
#define MySetupExeName "Obrew-Studio.WIN.Setup.exe"
#define MyComment "The tool for building Ai agents"

[Setup]
; NOTE: The value of AppId uniquely identifies this application. Do not use the same AppId value in installers for other applications.
; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
AppId={{3FFF4EFE-8FC9-4C78-A698-7A248A4BCADA}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
VersionInfoVersion={#MyAppVersion}
;AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={userappdata}\Obrew-Studio
DisableProgramGroupPage=yes
; the following line to run in non administrative install mode (install for current user only.)
PrivilegesRequired=lowest
OutputDir=C:\Project Files\brain-dump-ai\obrew-studio-server\installer
OutputBaseFilename=Obrew-Studio.WIN.Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "C:\Project Files\brain-dump-ai\obrew-studio-server\output\Obrew-Studio\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Project Files\brain-dump-ai\obrew-studio-server\output\Obrew-Studio\_deps\*"; DestDir: "{app}/_deps"; Flags: ignoreversion recursesubdirs createallsubdirs
; NOTE: Don't use "Flags: ignoreversion" on any shared system files

[Icons]
Name: "{userprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Comment: "{#MyComment}"
Name: "{userdesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Comment: "{#MyComment}"; Tasks: desktopicon
Name: "{userprograms}\{#MyAppName}-headless"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--headless=True"; Comment: "{#MyComment}"
Name: "{userdesktop}\{#MyAppName}-headless"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--headless=True"; Comment: "{#MyComment}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

