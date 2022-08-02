#include <tuple>
#include <map>
#include <iostream>
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
        std::string string;
        int number = 0;
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
            data.string = value;
            return *this;
        }
        
        Value& operator=(double value) {
            data.number = value;
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
        return value.data.string;
    }
    
    template<>
    std::string& asAny<std::string>(Value& value) {
        return value.data.string;
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
        return value.data.string;
    }
    
    template<>
    std::string getStr<std::string>(Value& value) {
        return value.data.string;
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
    if (Json::asAny<int>(data["type"]) == OPERATION) { // operator
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
    std::string data_ = "{\n";

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
    
    const char separator1 = ',';
    const char separator2 = '\n';
    const char separator3 = ':';

    std::vector<std::string> outputArray1;

    std::map<std::string, std::string> values;

    std::stringstream streamData1(data);
    std::string val, tok;
    std::pair<std::string, std::string> property_;

    while (std::getline(streamData1, val, separator1)) {
        outputArray1.push_back(val);
    }
    
    for (int i=0; i<outputArray1.size(); i++ ) {
        std::stringstream streamData2(outputArray1[i]);
        while (std::getline(streamData2, val, separator2)) {
            if (val != "{" and val != "}" and val.size() != 0) {
                std::stringstream streamData3(val);
                bool first=true;
                while (std::getline(streamData3, tok, separator3)) {
                    tok.erase(remove_if(tok.begin(), tok.end(), isspace), tok.end());
                    if(first) {
                        property_.first = tok;
                        first = false;
                    }
                    else {
                        property_.second = tok;
                    }
                }
                values.insert(property_);
            }
        }
    }

    constexpr auto nbProperties = std::tuple_size<decltype(T::properties_header)>::value;

    for_sequence(std::make_index_sequence<nbProperties>{}, [&](auto i){
        // get the property
        constexpr auto property = std::get<i>(T::properties_header);
        using Type = typename decltype(property)::Type; 
        // set the value to the member
        data_[property.name] = Json::getValue<Type>(values.find(property.name)->second);
        //std::cout << "--" << property.name << std::endl;
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
            //std::cout << "--" << property.name << std::endl;
            //std::cout << "--" << values.find(property.name)->second << std::endl;
        }); 
    }

    
   return data_;
}
