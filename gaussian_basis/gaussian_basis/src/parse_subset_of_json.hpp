#include <string>
#include <vector>

#ifndef _PARSE_SUBSET_OF_JSON_
#define _PARSE_SUBSET_OF_JSON_

using std::string;

namespace parse_subset_of_json {

    struct Value;

    struct KeyValuePair;

    class Bracket {
        std::vector<KeyValuePair> key_value_pairs;
        public:
        Bracket() {
            this->key_value_pairs = {};
        };
        std::vector<string> keys() const;
        Value operator[](const std::string &key) const;
        // Value& operator[](const std::string &key);
        std::string print() const;
        void add(const std::string &key, double number);
        void add(
            const std::string &key, 
            const std::vector<double> &number_list);
        void add(const std::string &key, const Bracket &bracket);
        void add(
            const std::string &key, const std::vector<Bracket> &bracket_list);
    };

    struct Value {
        double number;
        std::vector<double> number_list;
        Bracket bracket;
        std::vector<Bracket> bracket_list;
    };

    struct KeyValuePair {
        enum class Type {
            NUMBER=0, NUMBER_LIST=1, BRACKET=2, BRACKET_LIST=3};
        Type type;
        string key;
        Value value;
    };

    /* Parse an extremely simple subset of JSON, where the keys must be of type 
    string, and their corresponding values must either be numbers, 
    lists of only numbers, or lists containing only sub-objects with these 
    properties.

    Absolutely no checks are done to ensure that the above constraints
    are actually fulfilled in the input string,
    in which case this function may crash the program.
    */
    Bracket parse(const std::string &contents);

    void example1();
    void example2();
    void example3(const std::string &);

}

#endif
