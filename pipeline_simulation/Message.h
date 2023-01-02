#ifndef _MESSAGE_H_
#define _MESSAGE_H_

#define OPERATION 0
#define REFACTOR_COMPUTE_NODE 1
#define REFACTOR_DATA_OWNER 2


#include <tuple>
#include <map>
#include <string>
#include <iostream>
#include <sstream>
#include <sys/types.h> 
#include <stdlib.h>
#include <vector>

// sequence for
template <typename T, T... S, typename F>
constexpr void for_sequence(std::integer_sequence<T, S...>, F&& f) {
	using unpack_t = int[];
	(void)unpack_t{(static_cast<void>(f(std::integral_constant<T, S>{})), 0)..., 0};
}

// Sample implementation of a json-like data structure. It is only there for the example to compile and actually produce a testable output
namespace Json {
    struct Value;
    
    struct ValueData {
        std::map<std::string, Value> subObject;
        std::string string_;
        int number = 0;
        std::vector<int> numbers;
        std::vector<std::pair<int, std::string>> addr;
    };
    
    struct Value {
        ValueData data;
        
        Value& operator[](std::string name) {
            return data.subObject[std::move(name)];
        }
        
        const Value& operator[](std::string name) const {
            auto it = data.subObject.find(std::move(name));
            
            if (it != data.subObject.end()) {
                return it->second;
            }
            
            throw;
        }
        
        Value& operator=(std::string value) {
            data.string_ = value;
            return *this;
        }
        
        Value& operator=(double value) {
            data.number = value;
            return *this;
        }

        Value& operator=(std::vector<int> value) {
            data.numbers = value;
            return *this;
        }

        Value& operator=(std::vector<std::pair<int, std::string>> value) {
            data.addr = value;
            return *this;
        }
    };
    
    template<typename T> T& asAny(Value&);
    template<typename T> const T& asAny(const Value&);

    template<typename T> std::string getStr(Value&);
    template<typename T> const std::string getStr(const Value&);

    template<typename T> T getValue(std::string&);
    template<typename T> const T getValue(const std::string&);
    
    template<>
    int& asAny<int>(Value& value) {
        return value.data.number;
    }
    
    template<>
    const int& asAny<int>(const Value& value) {
        return value.data.number;
    }
    
    template<>
    const std::string& asAny<std::string>(const Value& value) {
        return value.data.string_;
    }
    
    template<>
    std::string& asAny<std::string>(Value& value) {
        return value.data.string_;
    }

    template<>
    const std::vector<int>& asAny<std::vector<int>>(const Value& value) {
        return value.data.numbers;
    }
    
    template<>
    std::vector<int>& asAny<std::vector<int>>(Value& value) {
        return value.data.numbers;
    }

    template<>
    const std::vector<std::pair<int, std::string>>& asAny<std::vector<std::pair<int, std::string>>>(const Value& value) {
        return value.data.addr;
    }
    
    template<>
    std::vector<std::pair<int, std::string>>& asAny<std::vector<std::pair<int, std::string>>>(Value& value) {
        return value.data.addr;
    }

    //..
    
    template<>
    std::string getStr<int>(Value& value) {
        auto s = std::to_string(value.data.number);
        return s;
    }
    
    template<>
    const std::string getStr<int>(const Value& value) {
        auto s = std::to_string(value.data.number);
        return s;
    }
    
    template<>
    const std::string getStr<std::string>(const Value& value) {
        return value.data.string_;
    }
    
    template<>
    std::string getStr<std::string>(Value& value) {
        return value.data.string_;
    }

    template<>
    const std::string getStr<std::vector<int>>(const Value& value) {
        std::string text = "[ ";

        for(int i=0; i<value.data.numbers.size(); i++) {
            text = text + std::to_string(value.data.numbers[i]) + " ";
        }
        text = text + "]";
        return text;
    }
    
    template<>
    std::string getStr<std::vector<int>>(Value& value) {
        std::string text = "[ ";

        for(int i=0; i<value.data.numbers.size(); i++) {
            text = text + std::to_string(value.data.numbers[i]) + " ";
        }
        text = text + "]";
        return text;
    }

    template<>
    const std::string getStr<std::vector<std::pair<int, std::string>>>(const Value& value) {
        std::string text = "[ ";

        for(int i=0; i<value.data.addr.size(); i++) {
            text = text + std::to_string(value.data.addr[i].first) + "," + value.data.addr[i].second + " ";
        }
        text = text + "]";
        return text;
    }
    
    template<>
    std::string getStr<std::vector<std::pair<int, std::string>>>(Value& value) {
        std::string text = "[ ";

        for(int i=0; i<value.data.addr.size(); i++) {
            text = text + std::to_string(value.data.addr[i].first) + "," + value.data.addr[i].second + " ";
        }
        text = text + "]";
        return text;
    }

    // ..

    template<>
    int getValue<int>(std::string& value) {
        auto s = stoi(value);
        return s;
    }

    template<>
    const int getValue<int>(const std::string& value) {
        auto s = stoi(value);
        return s;
    }

    template<>
    const std::string getValue<std::string>(const std::string& value) {
        return value;
    }

    template<>
    std::string getValue<std::string>(std::string& value) {
        return value;
    }

    template<>
    const std::vector<int> getValue<std::vector<int>>(const std::string& value) {
        std::vector<int> my_data;
        const char separator = ' ';
        std::stringstream streamData(value);
        std::string val;
        while (std::getline(streamData, val, separator)) {
            if (val != "[" && val != "]" && val != "") {
                my_data.push_back(stoi(val));
            }
            
        }
        return my_data;
    }

    template<>
    std::vector<int> getValue<std::vector<int>>(std::string& value) {
        std::vector<int> my_data;
        const char separator = ' ';
        std::stringstream streamData(value);
        std::string val;
        while (std::getline(streamData, val, separator)) {
            if (val != "[" && val != "]" && val != "") {
                my_data.push_back(stoi(val));
            }
            
        }
        return my_data;
    }

    template<>
    const std::vector<std::pair<int, std::string>> getValue<std::vector<std::pair<int, std::string>>>(const std::string& value) {
        std::vector<std::pair<int, std::string>> my_data;
        const char separator = ' ';
        const char separator_ = ',';
        std::stringstream streamData(value);
        std::string val;
        while (std::getline(streamData, val, separator)) {
            if (val != "[" && val != "]" && val != "") {
                std::stringstream streamValue(val);
                std::string val_;
                std::getline(streamValue, val_, separator_);
                int id = stoi(val_);
                std::getline(streamValue, val_, separator_);
                std::string ip = val_;
                my_data.push_back(std::pair<int, std::string>(id, ip));
            }
            
        }
        return my_data;
    }

    template<>
    std::vector<std::pair<int, std::string>> getValue<std::vector<std::pair<int, std::string>>>(std::string& value) {
        std::vector<std::pair<int, std::string>> my_data;
        const char separator = ' ';
        const char separator_ = ',';
        std::stringstream streamData(value);
        std::string val;
        while (std::getline(streamData, val, separator)) {
            if (val != "[" && val != "]" && val != "") {
                std::stringstream streamValue(val);
                std::string val_;
                std::getline(streamValue, val_, separator_);
                int id = stoi(val_);
                std::getline(streamValue, val_, separator_);
                std::string ip = val_;
                my_data.push_back(std::pair<int, std::string>(id, ip));
            }
            
        }
        return my_data;
    }
    
}

template<typename Class, typename T>
struct PropertyImpl {
    constexpr PropertyImpl(T Class::*aMember, const char* aName) : member{aMember}, name{aName} {}

    using Type = T;

    T Class::*member;
    const char* name;
};

// One could overload this function to accept both a getter and a setter instead of a member.
template<typename Class, typename T>
constexpr auto property(T Class::*member, const char* name) {
    return PropertyImpl<Class, T>{member, name};
}


// unserialize function
template<typename T>
T fromJson(const Json::Value& data) {
    T object;

    // We first get the number of properties
    constexpr auto nbProperties = std::tuple_size<decltype(T::properties_header)>::value;
    
    // We iterate on the index sequence of size `nbProperties`
    for_sequence(std::make_index_sequence<nbProperties>{}, [&](auto i){
        // get the property
        constexpr auto property = std::get<i>(T::properties_header);

        // get the type of the property
        using Type = typename decltype(property)::Type;

        // set the value to the member
        object.*(property.member) = Json::asAny<Type>(data[property.name]);

    });
    if (Json::asAny<int>(data["type"]) == OPERATION) { // operation
        constexpr auto nbProperties = std::tuple_size<decltype(T::properties_operator)>::value;
        // We iterate on the index sequence of size `nbProperties`
        for_sequence(std::make_index_sequence<nbProperties>{}, [&](auto i){
            // get the property
            constexpr auto property = std::get<i>(T::properties_operator);

            // get the type of the property
            using Type = typename decltype(property)::Type;

            // set the value to the member
            object.*(property.member) = Json::asAny<Type>(data[property.name]);

        });

    }
    else{
        constexpr auto nbProperties = std::tuple_size<decltype(T::properties_refactor)>::value;
        // We iterate on the index sequence of size `nbProperties`
        for_sequence(std::make_index_sequence<nbProperties>{}, [&](auto i){
            // get the property
            constexpr auto property = std::get<i>(T::properties_refactor);

            // get the type of the property
            using Type = typename decltype(property)::Type;
            // set the value to the member
            object.*(property.member) = Json::asAny<Type>(data[property.name]);

        });
    }

    return object;
}

template<typename T>
std::string fromJson_toStr(const Json::Value& data) {
    T object;
    std::string data_ = "{,\n";

    // We first get the number of properties
    constexpr auto nbProperties = std::tuple_size<decltype(T::properties_header)>::value;

    for_sequence(std::make_index_sequence<nbProperties>{}, [&](auto i){
        // get the property
        constexpr auto property = std::get<i>(T::properties_header);

        // get the type of the property
        using Type = typename decltype(property)::Type;

        // set the value to the member
        object.*(property.member) = Json::asAny<Type>(data[property.name]);
        data_ = data_ + property.name + " : " + Json::getStr<Type>(data[property.name]) + ",\n";
    });

    if (Json::asAny<int>(data["type"]) == OPERATION) { // operator
        constexpr auto nbProperties = std::tuple_size<decltype(T::properties_operator)>::value;
    
        for_sequence(std::make_index_sequence<nbProperties>{}, [&](auto i){
            // get the property
            constexpr auto property = std::get<i>(T::properties_operator);
            // get the type of the property
            using Type = typename decltype(property)::Type;

            // set the value to the member
            object.*(property.member) = Json::asAny<Type>(data[property.name]);
            data_ = data_ + property.name + " : " + Json::getStr<Type>(data[property.name]) + ",\n";
        }); 
    }
    else {
        constexpr auto nbProperties = std::tuple_size<decltype(T::properties_refactor)>::value;
    
        for_sequence(std::make_index_sequence<nbProperties>{}, [&](auto i){
            // get the property
            constexpr auto property = std::get<i>(T::properties_refactor);
            // get the type of the property
            using Type = typename decltype(property)::Type;

            // set the value to the member
            object.*(property.member) = Json::asAny<Type>(data[property.name]);
            data_ = data_ + property.name + " : " + Json::getStr<Type>(data[property.name]) + ",\n";
        }); 
    }
    data_ = data_ + "}";
    return data_;
}

template<typename T>
Json::Value toJson(const T& object) {
    Json::Value data;

    // We first get the number of properties
    constexpr auto nbProperties = std::tuple_size<decltype(T::properties_header)>::value;

    for_sequence(std::make_index_sequence<nbProperties>{}, [&](auto i){
        // get the property
        constexpr auto property = std::get<i>(T::properties_header);

        // set the value to the member
        data[property.name] = object.*(property.member);
    });
    
    if (Json::asAny<int>(data["type"]) == OPERATION) { // operator
        constexpr auto nbProperties = std::tuple_size<decltype(T::properties_operator)>::value;
    
        for_sequence(std::make_index_sequence<nbProperties>{}, [&](auto i){
            // get the property
            constexpr auto property = std::get<i>(T::properties_operator);

            // set the value to the member
            data[property.name] = object.*(property.member);
        }); 
    }
    else {
        constexpr auto nbProperties = std::tuple_size<decltype(T::properties_refactor)>::value;
    
        for_sequence(std::make_index_sequence<nbProperties>{}, [&](auto i){
            // get the property
            constexpr auto property = std::get<i>(T::properties_refactor);
            // set the value to the member
            data[property.name] = object.*(property.member);
        }); 
    }

    return data;
}

template<typename T>
Json::Value fromStr_toJson(const std::string& data) {
    Json::Value data_;
    std::string data_c = data;
    std::map<std::string, std::string> values;
    std::pair<std::string, std::string> property_;
    std::string delimiter = ",\n";
    std::string delimiter2 = " : ";
    size_t pos = 0, pos_=0;
    std::string token, tok;
    bool stop = false;
    std::string rec{"values"};
    while (!stop) {
        pos = data_c.find(delimiter);

        auto s1 = data_c.substr(0, rec.size());
        if(s1 == rec) {
            stop = true;
            token = data_c.substr(0, data_c.length()-3);
        }
        else if (pos == std::string::npos) {
            stop = true;
            token = data_c;
        }
        else{
            token = data_c.substr(0, pos);
            data_c.erase(0, pos + delimiter.length());
        }
        pos_ = 0;
        if (token != "{" and token != "}" and token.size() != 0) {
            pos_ = token.find(delimiter2);
            property_.first = token.substr(0, pos_);
            token.erase(0, pos_ + delimiter2.length());
            property_.second = token;
            values.insert(property_);
        }
    }
    constexpr auto nbProperties = std::tuple_size<decltype(T::properties_header)>::value;

    for_sequence(std::make_index_sequence<nbProperties>{}, [&](auto i){
        // get the property
        constexpr auto property = std::get<i>(T::properties_header);
        using Type = typename decltype(property)::Type; 
        // set the value to the member
        data_[property.name] = Json::getValue<Type>(values.find(property.name)->second);
    });
    
    if (Json::asAny<int>(data_["type"]) == OPERATION) { // operator
        constexpr auto nbProperties = std::tuple_size<decltype(T::properties_operator)>::value;
    
        for_sequence(std::make_index_sequence<nbProperties>{}, [&](auto i){
            // get the property
            constexpr auto property = std::get<i>(T::properties_operator);
            using Type = typename decltype(property)::Type;
            // set the value to the member
            data_[property.name] = Json::getValue<Type>(values.find(property.name)->second);
        }); 
    }
    else {
        constexpr auto nbProperties = std::tuple_size<decltype(T::properties_refactor)>::value;
    
        for_sequence(std::make_index_sequence<nbProperties>{}, [&](auto i){
            // get the property
            constexpr auto property = std::get<i>(T::properties_refactor);
            using Type = typename decltype(property)::Type;
            // set the value to the member
            data_[property.name] = Json::getValue<Type>(values.find(property.name)->second);
        }); 
    }
   return data_;
}

struct Message {
    int save_connection; 
    int type; //operation=0 or refacto=1,2
    int dest;
    int model_part;
    // refactor
    int start=-1, end=-1, prev=-1, next=-1, dataset=-1, num_classes=-1, model_name=-1, model_type=-1;
    std::vector<int> data_owners;
    std::vector<std::pair<int, std::string>> rooting_table; // id: ip --> ignoring port num
    int read_table=1;
    // operation
    int client_id=-1, prev_node=-1, size_=-1, type_op=-1;
    std::string values;

    constexpr static auto properties_header = std::make_tuple(
        property(&Message::save_connection, "save_connection"),
        property(&Message::type, "type")
    );

    constexpr static auto properties_refactor = std::make_tuple( // refactor message
        property(&Message::start, "start"),
        property(&Message::end, "end"),
        property(&Message::prev, "prev"),
        property(&Message::next, "next"),
        property(&Message::dataset, "dataset"),
        property(&Message::num_classes, "num_classes"),
        property(&Message::model_name, "model_name"),
        property(&Message::model_type, "model_type"),
        property(&Message::data_owners, "data_owners"),
        property(&Message::rooting_table, "rooting_table"),
        property(&Message::read_table, "read_table")
    );

    constexpr static auto properties_operator = std::make_tuple(
        property(&Message::client_id, "client_id"),
        property(&Message::prev_node, "prev_node"),
        property(&Message::size_, "size_"),
        property(&Message::type_op, "type_op"),
        property(&Message::values, "values"),
        property(&Message::values, "model_part")
    );
};

std::ostream& operator<<(std::ostream& os, const Message& dt) {
    if (dt.type != 0) {
        os << "start: " << dt.start << ", " << "end: " << dt.end << ", " <<  "prev: " << dt.prev << ", " << "next: " << dt.next << ", " << std::endl;
    }
    else {
        os << "client id: " << dt.client_id << ", " << "prev node: " << dt.prev_node << std::endl;
    }
    return os;
}

#endif