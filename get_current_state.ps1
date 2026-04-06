$cpu = (Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples.CookedValue
$totalMemMB = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1MB
$freeMemMB = (Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue
$usedMemMB = $totalMemMB - $freeMemMB
$memPercent = ($usedMemMB / $totalMemMB) * 100
Write-Host "CPU: $([math]::Round($cpu,1))%"
Write-Host "Memory: $([math]::Round($usedMemMB,0)) MB / $([math]::Round($totalMemMB,0)) MB ($([math]::Round($memPercent,0))%)"

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpuUtil = (nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    $gpuMem = (nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits) -split ',\s*'
    $usedVRAM = [int]$gpuMem[0]
    $totalVRAM = [int]$gpuMem[1]
    $vramPercent = [math]::Round(($usedVRAM / $totalVRAM) * 100)
    Write-Host "GPU: $gpuUtil% | VRAM: $usedVRAM MiB / $totalVRAM MiB ($vramPercent%)"
} else {
    Write-Host "GPU: N/A (nvidia-smi not found)"
}
