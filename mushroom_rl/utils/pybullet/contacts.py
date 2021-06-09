class ContactHelper(object):
    def __init__(self, client, contacts, model_map, link_map):
        self._client = client

        self._contact_list = list()
        self._contact_name_map = dict()
        self._computed_contacts = dict()

        for contact in contacts:
            name_1, name_2 = contact.split('<->')

            model_1, link_1 = self._get_link_ids(name_1, link_map, model_map)
            model_2, link_2 = self._get_link_ids(name_2, link_map, model_map)

            self._contact_name_map[contact] = (model_1, model_2, link_1, link_2)
            self._add_contact(model_1, model_2, link_1, link_2)

    def compute_contacts(self):
        if len(self._contact_list) == 0:
            return

        self._reset_computed_contacts()
        for contact in self._contact_list:
            model_a, model_b = contact[0]
            contacts_points = self._client.getContactPoints(model_a, model_b)

            for contact_p in contacts_points:
                involved_links = contact_p[3:5]
                if involved_links in contact[1]:
                    self._computed_contacts[model_a][model_b][contact_p[3]][contact_p[4]] = contact_p

    def get_contact(self, contact_name):
        model_1, model_2, link_1, link_2 = self._order_contact(*self._contact_name_map[contact_name])
        return self._computed_contacts[model_1][model_2][link_1][link_2]

    def _add_contact(self, model_1, model_2, link_1, link_2):
        model_a, model_b, link_a, link_b = self._order_contact(model_1, model_2, link_1, link_2)

        done = False
        for contact in self._contact_list:
            models = contact[0]
            links_list = contact[1]
            if models == (model_a, model_b):
                exists = False
                for links in links_list:
                    if (link_a, link_b) == links:
                        exists = True
                        break
                if not exists:
                    links_list.append((link_a, link_b))
                    done = True
                    break

        if not done:
            contact = ((model_a, model_b), [(link_a, link_b)])
            self._contact_list.append(contact)

    def _reset_computed_contacts(self):
        for contact in self._contact_name_map.values():
            model_a, model_b, link_a, link_b = self._order_contact(*contact)

            if model_a not in self._computed_contacts:
                self._computed_contacts[model_a] = dict()
            if model_b not in self._computed_contacts[model_a]:
                self._computed_contacts[model_a][model_b] = dict()
            if link_a not in self._computed_contacts[model_a][model_b]:
                self._computed_contacts[model_a][model_b][link_a] = dict()
            self._computed_contacts[model_a][model_b][link_a][link_b] = None

    @staticmethod
    def _get_link_ids(name, link_map, model_map):
        if name in link_map:
            return link_map[name]
        else:
            return model_map[name], -1

    @staticmethod
    def _order_contact(model_1, model_2, link_1, link_2):
        if model_1 < model_2:
            model_a = model_1
            link_a = link_1
            model_b = model_2
            link_b = link_2
        else:
            model_a = model_2
            link_a = link_2
            model_b = model_1
            link_b = link_1

        return model_a, model_b, link_a, link_b
