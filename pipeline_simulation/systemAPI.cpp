#include "systemAPI.h"

//template <typename T>
void systemAPI::init_state(model_name name, int model_, int num_class, int start, int end) {
    ModelPart part(name, model_, start, end, num_class);
    for (int i=0; i<clients.size() ; i++) {
        State client(clients[i], part.layers);
        
        clients_state.insert(std::pair<int, State>(clients[i], client));
    }
}

