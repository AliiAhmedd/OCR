# Runs every embedding experiment in configs/embeddings/ (progress.md row 44): for each variant, evaluate
# A_baseline / configuration 14 / configuration 15 on Azurite, then retune the thresholds for that embedding.
# Then writes one comparison table: data/runs/embedding_comparison.md (scripts/compare_embeddings.py).
#
#   .\scripts\run_embeddings.ps1                         # every variant
#   .\scripts\run_embeddings.ps1 dinov3_vitb16 dinov2_base   # only these
#
# Needs: the emulators up and seeded (README), data/ground_truth_plus.csv, a CUDA PyTorch for the GPU, and for the
# two DINOv3 variants a Hugging Face login that accepted the DINOv3 licence (huggingface-cli login).
param([string[]]$Variants)

# no $ErrorActionPreference = "Stop": Windows PowerShell 5.1 would stop on Python's log output (stderr)
$py = ".venv\Scripts\python.exe"
if (-not $env:AZURE_STORAGE_CONNECTION_STRING) {
    $env:AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
}
if (-not $Variants) {
    $Variants = Get-ChildItem configs\embeddings\*.yaml | Where-Object { $_.Name -notlike "*_grid.yaml" } |
        ForEach-Object { $_.BaseName }
}
& $py -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"

foreach ($v in $Variants) {
    Write-Host "`n=== ${v}: evaluate (A_baseline, configuration 14, configuration 15) ==="
    & $py -m id_classifier evaluate --config "configs/embeddings/$v.yaml"
    if ($LASTEXITCODE -ne 0) { Write-Warning "${v} evaluate failed, skipped"; continue }
    Write-Host "=== ${v}: threshold tune ==="
    & $py -m id_classifier tune-routing --grid "configs/embeddings/${v}_grid.yaml" --out "data/sweeps/embed_${v}"
    if ($LASTEXITCODE -ne 0) { Write-Warning "${v} tune failed" }
}
& $py scripts/compare_embeddings.py
