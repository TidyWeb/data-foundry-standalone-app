#!/usr/bin/env bash
# Adds "Data Foundry" to your Linux applications menu with its own icon.
# Then: open the menu, right-click Data Foundry, choose "Add to Favorites" / "Pin to taskbar".
# Only creates files in ~/.local/share; to remove it later, delete data-foundry.desktop there.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$APPS"
chmod +x "$HERE/start.sh"
cat > "$APPS/data-foundry.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Data Foundry
Comment=Generate coherent random practice datasets
Exec="$HERE/start.sh"
Path=$HERE
Icon=$HERE/icons/data-foundry.png
Terminal=true
Categories=Education;Development;
StartupWMClass=data-foundry
DESKTOP
chmod +x "$APPS/data-foundry.desktop"
command -v update-desktop-database >/dev/null && update-desktop-database "$APPS" 2>/dev/null || true
echo "Installed: $APPS/data-foundry.desktop"
echo "Find 'Data Foundry' in your applications menu and pin it to the taskbar."
