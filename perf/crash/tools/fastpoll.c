// 200 Hz PCI config-space poller. Logs transitions with microsecond timestamps.
// Watches: GPU vendor ids (33:00.0, cc:00.0), card upstream-port PRIMARY_BUS (31:00.0, ca:00.0),
// root-port AER UESta/CESta (30:02.0, c9:02.0 at 0x104/0x110), GPU hwmon power (every 1 s).
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
struct w { const char *name; const char *path; int off; int len; int fd; uint32_t last; };
static struct w ws[] = {
  {"gpu0_vid", "/sys/bus/pci/devices/0000:33:00.0/config", 0, 2, -1, 0},
  {"gpu1_vid", "/sys/bus/pci/devices/0000:cc:00.0/config", 0, 2, -1, 0},
  {"up0_bus",  "/sys/bus/pci/devices/0000:31:00.0/config", 0x18, 4, -1, 0},
  {"up1_bus",  "/sys/bus/pci/devices/0000:ca:00.0/config", 0x18, 4, -1, 0},
  {"dn0_vid",  "/sys/bus/pci/devices/0000:32:00.0/config", 0, 2, -1, 0},
  {"dn1_vid",  "/sys/bus/pci/devices/0000:cb:00.0/config", 0, 2, -1, 0},
  {"rp0_uesta","/sys/bus/pci/devices/0000:30:02.0/config", 0x104, 4, -1, 0},
  {"rp0_cesta","/sys/bus/pci/devices/0000:30:02.0/config", 0x110, 4, -1, 0},
  {"rp0_lnksta","/sys/bus/pci/devices/0000:30:02.0/config", 0x52, 2, -1, 0},
  {"rp1_uesta","/sys/bus/pci/devices/0000:c9:02.0/config", 0x104, 4, -1, 0},
  {"rp1_cesta","/sys/bus/pci/devices/0000:c9:02.0/config", 0x110, 4, -1, 0},
  {"rp1_lnksta","/sys/bus/pci/devices/0000:c9:02.0/config", 0x52, 2, -1, 0},
};
static const char *pw[2] = {"/sys/class/drm/card1/device/hwmon/hwmon1/power1_average", "/sys/class/drm/card2/device/hwmon/hwmon2/power1_average"};
static void ts(char *b) { struct timespec t; clock_gettime(CLOCK_REALTIME, &t); struct tm tm; localtime_r(&t.tv_sec, &tm); int n = strftime(b, 32, "%H:%M:%S", &tm); sprintf(b + n, ".%06ld", t.tv_nsec / 1000); }
int main(int argc, char **argv) {
  int n = sizeof(ws) / sizeof(ws[0]); char tb[40];
  for (int i = 0; i < n; i++) { ws[i].fd = open(ws[i].path, O_RDONLY); if (ws[i].fd < 0) { perror(ws[i].path); return 1; } }
  long iter = 0; int first = 1;
  for (;;) {
    for (int i = 0; i < n; i++) {
      uint32_t v = 0; if (pread(ws[i].fd, &v, ws[i].len, ws[i].off) != ws[i].len) v = 0xdeadbeef;
      if (first || v != ws[i].last) { ts(tb); printf("%s %-10s %08x -> %08x\n", tb, ws[i].name, ws[i].last, v); fflush(stdout); ws[i].last = v; }
    }
    first = 0;
    if (iter % 200 == 0) { // 1 Hz: power heartbeat
      char line[128] = ""; for (int k = 0; k < 2; k++) { char buf[32] = "NA"; int fd = open(pw[k], O_RDONLY); if (fd >= 0) { int r = read(fd, buf, 31); if (r > 0) buf[r - 1] = 0; close(fd); } long uw = atol(buf); char t[40]; snprintf(t, 40, " P%d=%ldW", k, uw / 1000000); strcat(line, t); }
      ts(tb); if (iter % 12000 == 0 || getenv("FASTPOLL_VERBOSE")) { printf("%s hb%s\n", tb, line); fflush(stdout); }
      // always write 1 Hz power to a side file for correlation
      FILE *f = fopen("/tmp/fastpoll_power.log", "a"); if (f) { fprintf(f, "%s%s\n", tb, line); fclose(f); }
    }
    iter++; usleep(5000);
  }
}
