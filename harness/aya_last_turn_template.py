"""Aya Expanse's own chat template with TRL generation tags around the FINAL assistant turn only (10 Nov 2026).
Renders byte-identical text to the stock template; with assistant_only_loss=True, TRL then puts loss on the target
reply only. Earlier assistant turns in our data are shipped Aya's replies (the behaviour being trained away), so
they must not carry loss. Same masking as the TRL LodeTuner recipe (completion_only_loss on the last reply)."""
TEMPLATE = (
    "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}"
    "{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}"
    "{% set system_message = false %}{% endif %}{% if system_message != false %}"
    "{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + system_message + '<|END_OF_TURN_TOKEN|>' }}{% endif %}"
    "{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}"
    "{% set content = message['content'] %}{% if message['role'] == 'user' %}"
    "{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}"
    "{% elif message['role'] == 'assistant' %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}"
    "{% if loop.last %}{% generation %}{{ content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endgeneration %}"
    "{% else %}{{ content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}")
