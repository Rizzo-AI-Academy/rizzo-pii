; Inno Setup script per l'installer di Anonimizzatore PII (build CPU).
; Genera l'installer con:  iscc installer.iss   (richiede Inno Setup: https://jrsoftware.org/isdl.php)
; Prerequisito: aver gia' eseguito la build PyInstaller (cartella dist\AnonimizzatorePII).

[Setup]
AppName=Anonimizzatore PII
AppVersion=1.0
AppPublisher=Rizzo AI
DefaultDirName={autopf}\AnonimizzatorePII
DefaultGroupName=Anonimizzatore PII
OutputDir=installer_out
OutputBaseFilename=AnonimizzatorePII-Setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest          ; installazione per-utente, niente admin

[Languages]
Name: "it"; MessagesFile: "compiler:Languages\Italian.isl"

[Tasks]
Name: "desktopicon"; Description: "Crea un'icona sul desktop"; GroupDescription: "Icone aggiuntive:"

[Files]
Source: "dist\AnonimizzatorePII\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\Anonimizzatore PII"; Filename: "{app}\AnonimizzatorePII.exe"
Name: "{group}\Disinstalla Anonimizzatore PII"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Anonimizzatore PII"; Filename: "{app}\AnonimizzatorePII.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\AnonimizzatorePII.exe"; Description: "Avvia Anonimizzatore PII"; Flags: nowait postinstall skipifsilent
