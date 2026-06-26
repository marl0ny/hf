#include "parse_subset_of_json.hpp"

#include <string>
#include <fstream>
#include <iostream>
#include <vector>

using namespace parse_subset_of_json;

static const string LETTERS 
    = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM";
static const string NUMERICAL_CHARS = "0123456789e-+.";

static std::string open_file(std::string filename) {
    std::fstream file{std::string(filename), std::fstream::in};
    if (!file.is_open()) {
        std::cerr << "Unable to open file " << filename << ".\n";
        return "";
    }
    std::string file_contents = "";
    char c;
    while ((c = file.get()) && !file.eof())
        file_contents += c;
    file.close();
    return file_contents;
}

static void write_file(std::string filename, std::string file_contents) {
    std::fstream file{std::string(filename), std::fstream::out};
    if (!file.is_open())
        std::cerr << "Unable to write to file " << filename << ".\n";
    file.write(&file_contents[0], file_contents.size());
    file.close();
}


static std::string print_number_list(std::vector<double> number_list) {
    std::string contents = "[\n";
    int num_list_size = number_list.size();
    for (int j = 0; j < num_list_size; j++) {
        double d = number_list[j];
        contents += "\t\t" + std::to_string(d);
        contents += (j == num_list_size - 1)? "": ",";
        contents += "\n";
    }
    contents += "\t]";
    return contents;
}

static std::string add_extra_tabs(
    std::string bracketed_value, 
    bool single_tab_for_end_line=true) {
    int new_line_total = 0;
    for (auto &c: bracketed_value)
        if (c == '\n')
            new_line_total++;
    std::string new_contents = "";
    int new_line_count = 0;
    for (auto &c: bracketed_value) {
        new_contents += c;
        if (c == '\n') {
            new_line_count++;
            if (new_line_count < new_line_total) {
                new_contents += "\t\t";
            } else if (new_line_count == new_line_total) {
                if (single_tab_for_end_line)
                    new_contents += "\t";
                else
                    new_contents += "\t\t";
            }
        }
    }
    return new_contents;
}

static std::string format_bracket_list(
    const std::vector<Bracket> &bracket_list
) {
    std::string contents = "[\n";
    int k = 0;
    for (auto const& bracket: bracket_list) {
        contents += "\t\t" 
            + add_extra_tabs(bracket.print(), false);
        contents += (k < bracket_list.size()-1)? ",":"";
        contents += "\n";
        k++;
    }
    contents += "\t]";
    return contents;
}

static std::string convert_tabs_to_four_spaces(std::string s) {
    std::string new_string = "";
    for (auto &e: s)
        if (e == '\t')
            new_string += "    ";
        else
            new_string += e;
    return new_string;
}

std::vector<string> Bracket::keys() const {
    std::vector<string> keys {};
    for (auto &key_value: this->key_value_pairs)
        keys.push_back(key_value.key);
    return keys;
}

Value Bracket::operator[](const std::string &key) const {
    for (auto &key_value: this->key_value_pairs)
        if (key_value.key == key)
            return key_value.value;
    return {};
}

// Value& Bracket::operator[](const std::string &key) {

// }

std::string Bracket::print() const {
    std::string contents = "{";
    for (int i = 0; i < this->key_value_pairs.size(); i++) {
        KeyValuePair pair = this->key_value_pairs[i];
        if (i == 0)
            contents += '\n';
        contents += "\t\"" + pair.key + "\": ";
        switch(pair.type) {
            case KeyValuePair::Type::NUMBER:
            contents += std::to_string(pair.value.number);
            break;
            case KeyValuePair::Type::NUMBER_LIST:
            contents += print_number_list(pair.value.number_list);
            break;
            case KeyValuePair::Type::BRACKET:
            contents += add_extra_tabs(pair.value.bracket.print());
            break;
            case KeyValuePair::Type::BRACKET_LIST:
            contents 
                += format_bracket_list(pair.value.bracket_list);
            break;
        }
        contents += (i == this->key_value_pairs.size() - 1)? "": ",";
        contents += "\n";
    }
    contents += "}";
    return convert_tabs_to_four_spaces(contents);
}

void Bracket::add(const std::string &key, double number) {
    struct KeyValuePair key_value_pair {
        .type=KeyValuePair::Type::NUMBER,
        .key=key,
        .value {
            .number=number,
        }
    };
    this->key_value_pairs.push_back(key_value_pair);
}

void Bracket::add(
    const std::string &key, const std::vector<double> &number_list) {
    struct KeyValuePair key_value_pair {
        .type=KeyValuePair::Type::NUMBER_LIST,
        .key=key,
        .value {
            .number_list=number_list,
        }
    };
    this->key_value_pairs.push_back(key_value_pair);
}

void Bracket::add(const std::string &key, const Bracket &bracket) {
    struct KeyValuePair key_value_pair {
        .type=KeyValuePair::Type::BRACKET,
        .key=key,
        .value {
            .bracket=bracket,
        }
    };
    this->key_value_pairs.push_back(key_value_pair);
}

void Bracket::add(
    const std::string &key, const std::vector<Bracket> &bracket_list) {
    struct KeyValuePair key_value_pair {
        .type=KeyValuePair::Type::BRACKET_LIST,
        .key=key,
        .value {
            .bracket_list=bracket_list,
        }
    };
    this->key_value_pairs.push_back(key_value_pair);
}


static std::string get_string(int &i, const std::string &contents) {
    std::string val = "";
    for (; contents[i] != '\"' && i < contents.size(); i++)
        val += contents[i];
    return val;
}

static bool is_number(char c) {
    for (auto &number: NUMERICAL_CHARS)
        if (c == number)
            return true;
    return false;
}

static double get_number(int &i, const std::string &contents) {
    std::string val = "";
    for (; is_number(contents[i]); i++)
        val += contents[i];
    return std::stod(val);
}

static std::vector<double> get_number_list(int &i, const std::string &contents) {
    if (contents[i] == '[')
        i++;
    std::vector<double> number_list {};
    while (contents[i] != ']') {
        char c = contents[i];
        if (is_number(c))
            number_list.push_back(get_number(i, contents));
        else
            i++;
    }
    return number_list;
}

static Bracket parse_(int &i, std::string contents);


static std::vector<Bracket> get_bracket_list(
    int &i, const std::string &contents) {
    if (contents[i] == '[')
        i++;
    std::vector<Bracket> bracket_list {};
    while (contents[i] != ']') {
        char c = contents[i];
        if (c == '{') {
            Bracket bracket = parse_(i, contents);
            bracket_list.push_back(bracket);
        } else {
            i++;
        }
    }
    return bracket_list;
}

static bool is_number_list(int i, const std::string &contents) {
    if (contents[i] == '[')
        i++;
    while (contents[i] != ']') {
        if (contents[i] == '{')
            return false;
        i++;
    }
    return true;
}

static Bracket parse_(int &i, std::string contents) {
    if (contents[i] == '{')
        i++;
    Bracket b{};
    std::vector<string> keys;
    while (i < contents.length()) {
        char c = contents[i];
        if (c == '"') {
            string key = get_string(++i, contents);
            keys.push_back(key);
            i++;
        } else if (is_number(c)) {
            double number = get_number(i, contents);
            string key = keys[keys.size() - 1];
            keys.pop_back();
            b.add(key, number);
        } else if (c == '{') {
            Bracket bracket = parse_(i, contents);
            string key = keys[keys.size() - 1];
            keys.pop_back();
            b.add(key, bracket);
        } else if (c == '}') {
            i++;
            return b;
        } else if (c == '[') {
            if (is_number_list(i, contents)) {
                std::vector<double> num_list
                            = get_number_list(i, contents);
                string key = keys[keys.size() - 1];
                keys.pop_back();
                b.add(key, num_list);
                i++;
            } else {
                std::vector<Bracket> bracket_list
                    = get_bracket_list(i, contents);
                string key = keys[keys.size() - 1];
                keys.pop_back();
                b.add(key, bracket_list);
            }
        } 
        else {
            i++;
        }
    }
    return b;
}

Bracket parse_subset_of_json::parse(const std::string &contents) {
    int i = 0;
    return parse_(i, contents);
}

void example1() {
    Bracket b{};
    b.add("coefficient", 1.0);
    b.add("values", {1.0, 2.0, 3.0, 4.0});
    Bracket b2{};
    b2.add("value", b);
    b2.add("subValue", b);
    b2.add("values", {b, b2});
    std::cout << b2.print() << std::endl;
    write_file("test.json", b2.print());
}

const std::string EXAMPLE2_STRING = R"({
    "number": 1.04525,
    "numbers": [
        1,
        2,
        3.9,
        4.0],
    "subOject": {
        "number": 12.0,
        "subSubObject": {
            "number": 11.0,
            "numbers": [
                1,
                2,
                3,
                4,
                5,
                6.0,
                7.123,
                9.99
            ],
            "moreNumbers": [
                1,
                2,
                3,
                4,
                5.0
            ]
        }
    }
})";

void example2() {
    string val = EXAMPLE2_STRING;
    int i = 0;
    Bracket b = parse_(i, val);
    std::cout << b.print() << std::endl;
}

void example3(const std::string &file_name) {
    string contents = open_file(file_name);
    Bracket b = parse(contents);
    std::cout << b.print() << std::endl;
}

