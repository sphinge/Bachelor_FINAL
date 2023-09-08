#include "net/af.h"
#include "net/ipv6/addr.h"
#include "net/sock/tcp.h"

uint8_t buf[128];
sock_tcp_t sock;

int main(void)
{
    int res;
    sock_tcp_ep_t remote = SOCK_IPV6_EP_ANY;

    remote.port = 12345;
    ipv6_addr_from_str((ipv6_addr_t *)&remote.addr,
                       "fe80::8c7e:c5ff:fe33:abf2%tap0");
    if (sock_tcp_connect(&sock, &remote, 0, 0) < 0) {
        puts("Error connecting sock");
        return 1;
    }
    puts("Sending \"Hello!\"");
    if ((res = sock_tcp_write(&sock, "1", sizeof("1"))) < 0) {
        puts("Errored on write");
    }
    else {
        if ((res = sock_tcp_read(&sock, &buf, sizeof(buf),
                                 SOCK_NO_TIMEOUT)) <= 0) {
            puts("Disconnected");
        }
        printf("Read: \"");
        for (int i = 0; i < res; i++) {
            printf("%c", buf[i]);
        }
        puts("\"");
    }
    sock_tcp_disconnect(&sock);
    return res;
}