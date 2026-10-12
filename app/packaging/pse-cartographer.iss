#define AppName "PSE Cartographer"
[Setup]
AppId={{64299183-02b4-4f3c-ab0f-2c84d924f6c4}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Andrej Ristic / Zerava Technologies Inc.
AppPublisherURL=https://github.com/andrejristic/pse-cartographer
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile={#SourceDir}\LICENSE.txt
OutputDir={#OutDir}
OutputBaseFilename=PSECartographer-Setup-{#AppVersion}
SetupIconFile={#SourceDir}\pse-cartographer.ico
UninstallDisplayIcon={app}\pse-cartographer.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\pse-cartographer.exe"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\pse-cartographer.exe"; Tasks: desktopicon
[Run]
Filename: "{app}\pse-cartographer.exe"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
