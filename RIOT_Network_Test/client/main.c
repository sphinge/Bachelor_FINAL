#include "net/gnrc/tcp.h"

#define SERVER_ADDR "fe80::abcd" // Replace with the server's IPv6 address
#define SERVER_PORT 8080
#define BUFFER_SIZE 128

int main(void)
{
    gnrc_tcp_tcb_t tcb;
    gnrc_tcp_tcb_init(&tcb);

    gnrc_tcp_ep_t remote;
    gnrc_tcp_ep_from_str(&remote, "[fe80::abcd%5]:8080");

    if (gnrc_tcp_open(&tcb, &remote, 0) == 0) {
        char buffer[BUFFER_SIZE] = "Hello Server!";
        gnrc_tcp_send(&tcb, buffer, sizeof(buffer), GNRC_TCP_NO_TIMEOUT);

        gnrc_tcp_close(&tcb);
    }

    return 0;
}
