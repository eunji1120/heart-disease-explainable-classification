#!/bin/bash
# Uninstall Oracle MySQL (the .pkg-installed copy at /usr/local/mysql).
# Leaves the Homebrew MySQL install at /usr/local/opt/mysql/ untouched.
#
# Run this with:   bash scripts/uninstall_oracle_mysql.sh
# You will be asked for your macOS user password once (for sudo).

set -e

echo "==> Caching sudo credentials..."
sudo -v

echo ""
echo "==> Stopping Oracle MySQL LaunchDaemon..."
sudo launchctl unload /Library/LaunchDaemons/com.oracle.oss.mysql.mysqld.plist 2>/dev/null || true

echo "==> Removing Oracle LaunchDaemon plist..."
sudo rm -f /Library/LaunchDaemons/com.oracle.oss.mysql.mysqld.plist

echo "==> Killing any remaining Oracle mysqld processes..."
sudo pkill -f "/usr/local/mysql/bin/mysqld" 2>/dev/null || true
sleep 2

echo "==> Removing Oracle MySQL install directory (/usr/local/mysql)..."
sudo rm -rf /usr/local/mysql

echo "==> Removing any old versioned Oracle directories (/usr/local/mysql-*)..."
sudo rm -rf /usr/local/mysql-* 2>/dev/null || true

echo "==> Removing Oracle MySQL preference pane and startup items (if present)..."
sudo rm -rf /Library/PreferencePanes/MySQL.prefPane 2>/dev/null || true
sudo rm -rf "$HOME/Library/PreferencePanes/MySQL.prefPane" 2>/dev/null || true
sudo rm -rf /Library/StartupItems/MySQLCOM 2>/dev/null || true

echo "==> Cleaning stale Unix socket (so Homebrew mysqld can claim it)..."
sudo rm -f /tmp/mysql.sock

echo ""
echo "==> Verifying Oracle MySQL is gone..."
if [ -d /usr/local/mysql ]; then
  echo "WARNING: /usr/local/mysql still exists"
else
  echo "OK: /usr/local/mysql is gone"
fi

echo ""
echo "==> Remaining mysqld processes (should be none, or Homebrew only):"
ps aux | grep mysqld | grep -v grep || echo "(no mysqld processes running)"

echo ""
echo "==> Done. Next step: reset the Homebrew MySQL root password."
