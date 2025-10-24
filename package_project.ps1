# package_project.ps1 - Create a zip package of the project (ignores .git by default)
$Out = Join-Path -Path (Get-Location).Parent -ChildPath "temporal_forge_package.zip"
Write-Host "Creating package at: $Out"
Compress-Archive -Path * -DestinationPath $Out -Force
Write-Host "Done. Package created: $Out"