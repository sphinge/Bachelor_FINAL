#include "net/gnrc/tcp.h"

#define SERVER_PORT 8080
#define BUFFER_SIZE 128

int main(void)
{
    gnrc_tcp_tcb_t tcb;
    gnrc_tcp_tcb_init(&tcb);

    gnrc_tcp_ep_t local = GNRC_TCP_EP_ANY;
    local.port = SERVER_PORT;

    gnrc_tcp_tcb_queue_t tcb_queue; // The queue for incoming connections

    // Then call the function
    gnrc_tcp_listen(&tcb_queue, &tcb, 1, &local);

    //gnrc_tcp_listen(NULL, &tcb, &local);

    while (1) {
        gnrc_tcp_tcb_t conn_tcb;
        gnrc_tcp_tcb_init(&conn_tcb);

        if (gnrc_tcp_accept(NULL, &conn_tcb, GNRC_TCP_NO_TIMEOUT) == 0) {
            char buffer[BUFFER_SIZE];
            ssize_t rcvd_size = gnrc_tcp_recv(&conn_tcb, buffer, BUFFER_SIZE, GNRC_TCP_NO_TIMEOUT);

            if (rcvd_size > 0) {
                puts(buffer);
            }

            gnrc_tcp_close(&conn_tcb);
        }
    }

    return 0;
}
