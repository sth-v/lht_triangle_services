

#        |     No ENTRYPOINT      |	 ENTRYPOINT exec_entry p1_entry	 | ENTRYPOINT [“exec_entry”, “p1_entry”]          |
# -------+------------------------+----------------------------------+------------------------------------------------+
# No CMD | error, not allowed	  | /bin/sh -c exec_entry p1_entry   | exec_entry p1_entry                            |
# -------+------------------------+----------------------------------+------------------------------------------------+
# CMD    | [“exec_cmd”, “p1_cmd”] | /bin/sh -c exec_entry p1_entry   | exec_entry p1_entry exec_cmd p1_cmd            |
# -------+------------------------+----------------------------------+------------------------------------------------+
# CMD    | exec_cmd p1_cmd	      | /bin/sh -c exec_entry p1_entry   | exec_entry p1_entry /bin/sh -c exec_cmd p1_cmd |


FROM mambaorg/micromamba
USER root




# Для выполнения директивы ниже вам необходимо указать `syntax=docker/dockerfile:1` в начале файла

# 🐍 Setup micromamba.
# ⚙️ Source: https://hub.docker.com/r/mambaorg/micromamba
COPY --chown=root:root env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
RUN apt update
RUN apt install -y git python3 python3-dev python3-setuptools cmake gcc g++
# 🐳 Setting pre-build params and environment variables.
# ⚙️ Please set you environment globals :)
# ENV PARAM=value


RUN /usr/local/bin/_entrypoint.sh python -m pip install git+https://github.com/contextmachine/mmcore.git compute_rhino3d && \
    /usr/local/bin/_entrypoint.sh python -m pip install git+https://github.com/contextmachine/cxmdata.git && \
    /usr/local/bin/_entrypoint.sh python -m pip install git+https://github.com/sth-v/cxm_boto_client.git && \


ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]