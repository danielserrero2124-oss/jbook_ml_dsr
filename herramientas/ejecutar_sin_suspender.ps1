# Ejecuta herramientas/ejecutar.sh impidiendo que Windows suspenda el equipo mientras corre.
# La petición (SetThreadExecutionState) es temporal: se libera sola cuando termina este proceso.
# Uso: powershell -File herramientas\ejecutar_sin_suspender.ps1 03 04 05 06 07
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Notebooks)  # Prefijos de los notebooks a ejecutar

Add-Type -Namespace Win32 -Name Energia -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@  # Declara la función de la API de Windows que controla la suspensión

$ES_CONTINUOUS = [uint32]"0x80000000"  # La petición dura hasta que se cancele o termine el proceso
$ES_SYSTEM_REQUIRED = [uint32]"0x00000001"  # Pide que el sistema no entre en suspensión
[Win32.Energia]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null  # Activa la petición

$raiz = Split-Path -Parent $PSScriptRoot  # Carpeta raíz del proyecto
$bash = "C:\Program Files\Git\bin\bash.exe"  # Intérprete de bash (Git for Windows)
try {
    & $bash "$raiz/herramientas/ejecutar.sh" @Notebooks  # Construye y ejecuta los notebooks en orden
    $codigo = $LASTEXITCODE  # Código de salida de la cadena
} finally {
    [Win32.Energia]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null  # Libera la petición de no suspender
}
exit $codigo  # Propaga el código de salida
