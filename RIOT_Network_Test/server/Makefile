# Name of your application
APPLICATION = server

# If no BOARD is found in the environment, use this default:
BOARD ?= native

# This has to be the absolute path to the RIOT base directory:
RIOTBASE ?= /home/wiktoria/Desktop/Thesis/RIOT

USEMODULE += netdev_default

USEMODULE += auto_init_gnrc_netif

USEMODULE += gnrc_ipv6_default

USEMODULE += gnrc_ipv6_router_default

USEMODULE += gnrc_ipv6

USEMODULE += gnrc_icmpv6_echo

USEMODULE += gnrc_tcp

USEMODULE += gnrc_sock_tcp

USEMODULE += gnrc_rpl


# Change this to 0 show compiler invocation lines by default:
QUIET ?= 1

include $(RIOTBASE)/Makefile.include
