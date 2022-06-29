use std::collections::HashMap;
use std::collections::VecDeque;
use std::env;
use std::fs;

// TODO: make this an argument, maybe -debug
const TRACING: bool = false;

type IntSize = i32;

#[derive(Debug, Clone, PartialEq)]
enum Direction {
    Left,
    Right,
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Trace,
    PrettyPrint,
    Exit,
    Label(String),
    LabelPointer(Direction, String),
    Int(Direction, IntSize),
    Add(Direction),
    Sub(Direction),
    Dup(Direction),
    Swap(Direction),
    Move(Direction),
    Over(Direction),
    Drop(Direction),
    Shr(Direction),
    Shl(Direction),
    Or(Direction),
    And(Direction),
    Eq(Direction),
    Gt(Direction),
    Lt(Direction),
    GtEq(Direction),
    LtEq(Direction),
    Jmp(Direction),
    JmpIf(Direction),
}

#[derive(Default)]
struct Tokenizer {}

impl Tokenizer {
    fn parse(&self, input: String) -> Vec<Token> {
        let mut tokens: Vec<Token> = vec![];

        fn dir_token(
            direction: Option<Direction>,
            token: fn(Direction) -> Token,
            word: &str,
            il: usize,
            iw: usize,
        ) -> Token {
            match direction {
                Some(dir) => token(dir),
                _ => panic!("Direction missing for word `{}` at {}:{}", word, il, iw),
            }
        }

        for (il, line) in input.lines().enumerate() {
            if line.trim_start().starts_with("#") {
                continue;
            }

            for (iw, word) in line.split_whitespace().enumerate() {
                let mut chars = word.chars();
                let direction: Option<Direction> = match word {
                    word if word.starts_with("!") => {
                        chars.next();
                        Some(Direction::Left)
                    }
                    word if word.ends_with("!") => {
                        chars.next_back();
                        Some(Direction::Right)
                    }
                    word if word.ends_with(":") => {
                        chars.next_back();
                        tokens.push(Token::Label(chars.as_str().into()));
                        continue;
                    }
                    _ => None,
                };

                match chars.as_str() {
                    "trace" => tokens.push(Token::Trace),
                    "pprint" => tokens.push(Token::PrettyPrint),
                    "exit" => tokens.push(Token::Exit),
                    "add" => tokens.push(dir_token(direction, Token::Add, word, il, iw)),
                    "sub" => tokens.push(dir_token(direction, Token::Sub, word, il, iw)),
                    "dup" => tokens.push(dir_token(direction, Token::Dup, word, il, iw)),
                    "swap" => tokens.push(dir_token(direction, Token::Swap, word, il, iw)),
                    "move" => tokens.push(dir_token(direction, Token::Move, word, il, iw)),
                    "over" => tokens.push(dir_token(direction, Token::Over, word, il, iw)),
                    "drop" => tokens.push(dir_token(direction, Token::Drop, word, il, iw)),
                    "shr" => tokens.push(dir_token(direction, Token::Shr, word, il, iw)),
                    "shl" => tokens.push(dir_token(direction, Token::Shl, word, il, iw)),
                    "or" => tokens.push(dir_token(direction, Token::Or, word, il, iw)),
                    "and" => tokens.push(dir_token(direction, Token::And, word, il, iw)),
                    "eq" => tokens.push(dir_token(direction, Token::Eq, word, il, iw)),
                    ">" => tokens.push(dir_token(direction, Token::Gt, word, il, iw)),
                    "<" => tokens.push(dir_token(direction, Token::Lt, word, il, iw)),
                    ">=" => tokens.push(dir_token(direction, Token::GtEq, word, il, iw)),
                    "<=" => tokens.push(dir_token(direction, Token::LtEq, word, il, iw)),
                    "jmp" => tokens.push(dir_token(direction, Token::Jmp, word, il, iw)),
                    "jmpif" => tokens.push(dir_token(direction, Token::JmpIf, word, il, iw)),
                    value => match direction {
                        Some(dir) => {
                            let parsed_value = value.parse::<IntSize>();
                            match parsed_value {
                                Ok(num) => tokens.push(Token::Int(dir, num)),
                                _ => tokens.push(Token::LabelPointer(dir, value.into())),
                            }
                        }
                        _ => panic!("Direction missing for word `{}` at {}:{}", word, il, iw),
                    },
                }
            }
        }

        tokens
    }
}

struct Evaluator {
    labels: HashMap<String, usize>,
    data: VecDeque<IntSize>,
}

impl Evaluator {
    pub fn new() -> Self {
        Self {
            labels: HashMap::new(),
            data: VecDeque::new(),
        }
    }

    fn pop_data(&mut self, dir: &Direction) -> IntSize {
        let value = match dir {
            Direction::Left => self.data.pop_front(),
            Direction::Right => self.data.pop_back(),
        };
        value.unwrap_or_else(|| panic!("No value found on the {:?}", dir))
    }

    fn push_data(&mut self, dir: &Direction, value: IntSize) {
        match dir {
            Direction::Left => self.data.push_front(value),
            Direction::Right => self.data.push_back(value),
        }
    }

    fn evaluate(&mut self, tokens: Vec<Token>) {
        // store all label positions
        for (i, token) in tokens.iter().enumerate() {
            match token {
                Token::Label(name) => {
                    self.labels.insert(name.into(), i);
                }
                _ => continue,
            }
        }

        let mut ip = 0;
        while ip < tokens.len() {
            let token = &tokens[ip];
            if TRACING {
                println!("{ip}: {token:?} <- {:?}", self.data);
            }

            match token {
                Token::Trace => {
                    println!("{:?}", self.data)
                }
                Token::PrettyPrint => {
                    println!(
                        "{}",
                        self.data.iter().fold(String::new(), |acc, &n| acc
                            + (if n == 1 { "*" } else { " " }))
                    )
                }
                Token::Exit => return,
                Token::Int(dir, num) => self.push_data(&dir, *num),
                Token::Add(dir) => {
                    let a = self.pop_data(&dir);
                    let b = self.pop_data(&dir);
                    self.push_data(&dir, a + b);
                }
                Token::Sub(dir) => {
                    let a = self.pop_data(&dir);
                    let b = self.pop_data(&dir);
                    self.push_data(&dir, b - a);
                }
                Token::Dup(dir) => {
                    let a = self.pop_data(&dir);
                    self.push_data(&dir, a);
                    self.push_data(&dir, a);
                }
                Token::Swap(dir) => {
                    let a = self.pop_data(&dir);
                    let b = self.pop_data(&dir);
                    self.push_data(&dir, a);
                    self.push_data(&dir, b);
                }
                Token::Move(dir) => {
                    let a = self.pop_data(&dir);
                    let swapped_dir = match dir {
                        Direction::Left => Direction::Right,
                        Direction::Right => Direction::Left,
                    };
                    self.push_data(&swapped_dir, a)
                }
                Token::Over(dir) => {
                    let a = self.pop_data(&dir);
                    let b = self.pop_data(&dir);
                    self.push_data(&dir, b);
                    self.push_data(&dir, a);
                    self.push_data(&dir, b);
                }
                Token::Drop(dir) => {
                    self.pop_data(&dir);
                }
                Token::Shr(dir) => {
                    let a = self.pop_data(&dir);
                    let b = self.pop_data(&dir);
                    self.push_data(&dir, b >> a);
                }
                Token::Shl(dir) => {
                    let a = self.pop_data(&dir);
                    let b = self.pop_data(&dir);
                    self.push_data(&dir, b << a);
                }
                Token::Or(dir) => {
                    let a = self.pop_data(&dir);
                    let b = self.pop_data(&dir);
                    self.push_data(&dir, b | a);
                }
                Token::And(dir) => {
                    let a = self.pop_data(&dir);
                    let b = self.pop_data(&dir);
                    self.push_data(&dir, b & a);
                }
                Token::Eq(dir) => {
                    let a = self.pop_data(&dir);
                    let b = self.pop_data(&dir);
                    self.push_data(&dir, (a == b) as IntSize);
                }
                Token::Gt(dir) => {
                    let a = self.pop_data(&dir);
                    let b = self.pop_data(&dir);
                    self.push_data(&dir, (a > b) as IntSize);
                }
                Token::Lt(dir) => {
                    let a = self.pop_data(&dir);
                    let b = self.pop_data(&dir);
                    self.push_data(&dir, (a < b) as IntSize);
                }
                Token::GtEq(dir) => {
                    let a = self.pop_data(&dir);
                    let b = self.pop_data(&dir);
                    self.push_data(&dir, (a >= b) as IntSize);
                }
                Token::LtEq(dir) => {
                    let a = self.pop_data(&dir);
                    let b = self.pop_data(&dir);
                    self.push_data(&dir, (a <= b) as IntSize);
                }
                Token::Jmp(dir) => {
                    let addr = self.pop_data(&dir) as usize;
                    ip = addr - 1;
                }
                Token::JmpIf(dir) => {
                    let addr = self.pop_data(&dir) as usize;
                    let cond = self.pop_data(&dir) != 0;
                    if cond {
                        ip = addr - 1;
                    }
                }
                Token::Label(_) => {}
                Token::LabelPointer(dir, name) => match self.labels.get(name) {
                    Some(value) => {
                        self.push_data(&dir, *value as i32);
                    }
                    _ => panic!("Unknown label `{}`", name),
                },
            }

            ip += 1;
        }

        // println!("{:?}", self.data);
    }
}

#[test]
fn test_tokenize_int() {
    let input = "1! !2";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    assert_eq!(
        tokens,
        vec![
            Token::Int(Direction::Right, 1),
            Token::Int(Direction::Left, 2)
        ]
    );
}

#[test]
fn test_tokenize_add() {
    let input = "add! !add";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    assert_eq!(
        tokens,
        vec![Token::Add(Direction::Right), Token::Add(Direction::Left)]
    );
}

#[test]
fn test_tokenize_sub() {
    let input = "sub! !sub";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    assert_eq!(
        tokens,
        vec![Token::Sub(Direction::Right), Token::Sub(Direction::Left)]
    );
}

#[test]
fn test_tokenize_dup() {
    let input = "dup! !dup";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    assert_eq!(
        tokens,
        vec![Token::Dup(Direction::Right), Token::Dup(Direction::Left)]
    );
}

#[test]
fn test_tokenize_swap() {
    let input = "swap! !swap";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    assert_eq!(
        tokens,
        vec![Token::Swap(Direction::Right), Token::Swap(Direction::Left)]
    );
}

#[test]
fn test_tokenize_move() {
    let input = "move! !move";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    assert_eq!(
        tokens,
        vec![Token::Move(Direction::Right), Token::Move(Direction::Left)]
    );
}

#[test]
fn test_tokenize_over() {
    let input = "over! !over";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    assert_eq!(
        tokens,
        vec![Token::Over(Direction::Right), Token::Over(Direction::Left)]
    );
}

#[test]
fn test_tokenize_drop() {
    let input = "drop! !drop";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    assert_eq!(
        tokens,
        vec![Token::Drop(Direction::Right), Token::Drop(Direction::Left)]
    );
}

#[test]
fn test_tokenize_shr() {
    let input = "shr! !shr";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    assert_eq!(
        tokens,
        vec![Token::Shr(Direction::Right), Token::Shr(Direction::Left)]
    );
}

#[test]
fn test_tokenize_shl() {
    let input = "shl! !shl";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    assert_eq!(
        tokens,
        vec![Token::Shl(Direction::Right), Token::Shl(Direction::Left)]
    );
}

#[test]
fn test_tokenize_or() {
    let input = "or! !or";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    assert_eq!(
        tokens,
        vec![Token::Or(Direction::Right), Token::Or(Direction::Left)]
    );
}

#[test]
fn test_tokenize_and() {
    let input = "and! !and";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    assert_eq!(
        tokens,
        vec![Token::And(Direction::Right), Token::And(Direction::Left)]
    );
}

#[test]
fn test_tokenize_eq() {
    let input = "eq! !eq";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    assert_eq!(
        tokens,
        vec![Token::Eq(Direction::Right), Token::Eq(Direction::Left)]
    );
}

#[test]
fn test_tokenize_gt() {
    let input = ">! !>";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    assert_eq!(
        tokens,
        vec![Token::Gt(Direction::Right), Token::Gt(Direction::Left)]
    );
}

#[test]
fn test_tokenize_lt() {
    let input = "<! !<";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    assert_eq!(
        tokens,
        vec![Token::Lt(Direction::Right), Token::Lt(Direction::Left)]
    );
}

#[test]
fn test_tokenize_gteq() {
    let input = ">=! !>=";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    assert_eq!(
        tokens,
        vec![Token::GtEq(Direction::Right), Token::GtEq(Direction::Left)]
    );
}

#[test]
fn test_tokenize_lteq() {
    let input = "<=! !<=";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    assert_eq!(
        tokens,
        vec![Token::LtEq(Direction::Right), Token::LtEq(Direction::Left)]
    );
}

#[test]
fn test_tokenize_jmp() {
    let input = "jmp! !jmp";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    assert_eq!(
        tokens,
        vec![Token::Jmp(Direction::Right), Token::Jmp(Direction::Left)]
    );
}

#[test]
fn test_tokenize_jmpif() {
    let input = "jmpif! !jmpif";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    assert_eq!(
        tokens,
        vec![
            Token::JmpIf(Direction::Right),
            Token::JmpIf(Direction::Left)
        ]
    );
}

#[test]
fn test_tokenize_labels() {
    let input = "end: !end end!";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    assert_eq!(
        tokens,
        vec![
            Token::Label("end".into()),
            Token::LabelPointer(Direction::Left, "end".into()),
            Token::LabelPointer(Direction::Right, "end".into())
        ]
    );
}

#[test]
fn test_tokenize_trace() {
    let input = "trace !trace trace!";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());
    assert_eq!(tokens, vec![Token::Trace, Token::Trace, Token::Trace]);
}

#[test]
fn test_tokenize_pprint() {
    let input = "pprint !pprint pprint!";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());
    assert_eq!(
        tokens,
        vec![Token::PrettyPrint, Token::PrettyPrint, Token::PrettyPrint]
    );
}

#[test]
fn test_tokenize_exit() {
    let input = "exit !exit exit!";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());
    assert_eq!(tokens, vec![Token::Exit, Token::Exit, Token::Exit]);
}

// #[test]
// fn test_evaluator_email_example() {
//     let input = "3! !5 !2 sub! !add";
//     let tokenizer = Tokenizer::default();
//     let tokens = tokenizer.parse(input.into());

//     let mut evaluator = Evaluator::new();
//     evaluator.evaluate(tokens);
//     assert_eq!(evaluator.data.get(0), Some(&4));
// }

#[test]
fn test_evaluator_int() {
    let input = "1! 2! !3";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![3, 1, 2]);
}

#[test]
fn test_evaluator_add() {
    let input = "1! 3! 2! add!";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data.get(1), Some(&5));

    let input = "1! 3! 2! !add";
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data.get(0), Some(&4));
}

#[test]
fn test_evaluator_sub() {
    let input = "1! 3! 2! sub!";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data.get(1), Some(&1));

    let input = "1! 3! 2! !sub";
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data.get(0), Some(&2));
}

#[test]
fn test_evaluator_dup() {
    let input = "1! 2! dup!";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![1, 2, 2]);

    let input = "1! 2! !dup";
    let tokens = tokenizer.parse(input.into());
    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![1, 1, 2]);
}

#[test]
fn test_evaluator_swap() {
    let input = "1! 2! 3! swap!";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![1, 3, 2]);

    let input = "1! 2! 3! !swap";
    let tokens = tokenizer.parse(input.into());
    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![2, 1, 3]);
}

#[test]
fn test_evaluator_move() {
    let input = "1! 2! 3! move!";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![3, 1, 2]);

    let input = "1! 2! 3! !move";
    let tokens = tokenizer.parse(input.into());
    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![2, 3, 1]);
}

#[test]
fn test_evaluator_over() {
    let input = "1! 2! over!";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![1, 2, 1]);

    let input = "1! 2! !over";
    let tokens = tokenizer.parse(input.into());
    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![2, 1, 2]);
}

#[test]
fn test_evaluator_drop() {
    let input = "1! 2! drop!";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![1]);

    let input = "1! 2! !drop";
    let tokens = tokenizer.parse(input.into());
    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![2]);
}

#[test]
fn test_evaluator_shr() {
    let input = "4! 1! shr!";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![2]);

    let input = "1! 4! !shr";
    let tokens = tokenizer.parse(input.into());
    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![2]);
}

#[test]
fn test_evaluator_shl() {
    let input = "4! 1! shl!";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![8]);

    let input = "1! 4! !shl";
    let tokens = tokenizer.parse(input.into());
    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![8]);
}

#[test]
fn test_evaluator_or() {
    let input = "4! 2! or!";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![6]);
}

#[test]
fn test_evaluator_and() {
    let input = "6! 2! and!";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![2]);
}

#[test]
fn test_evaluator_eq() {
    let input = "1! 1! 3! eq!";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![1, 0]);

    let input = "1! 3! 3! eq!";
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![1, 1]);

    let input = "1! 1! 3! !eq";
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![1, 3]);

    let input = "1! 3! 3! !eq";
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![0, 3]);
}

#[test]
fn test_evaluator_gt() {
    let input = "6! 2! >!";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![0]);

    let input = "2! 6! >!";
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![1]);

    let input = "6! 2! !>";
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![1]);

    let input = "2! 6! !>";
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![0]);
}

#[test]
fn test_evaluator_lt() {
    let input = "6! 2! <!";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![1]);

    let input = "2! 6! <!";
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![0]);

    let input = "6! 2! !<";
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![0]);

    let input = "2! 6! !<";
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![1]);
}

#[test]
fn test_evaluator_gteq() {
    let input = "6! 5! >=!";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![0]);

    let input = "6! 6! >=!";
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![1]);

    let input = "6! 6! !>=";
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![1]);

    let input = "5! 6! !>=";
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![0]);
}

#[test]
fn test_evaluator_lteq() {
    let input = "6! 6! <=!";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![1]);

    let input = "5! 6! <=!";
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![0]);

    let input = "6! 5! !<=";
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![0]);

    let input = "6! 6! !<=";
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![1]);
}

#[test]
fn test_evaluator_jmp() {
    let input = "1! 4! jmp! 2! 3!";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![1, 3]);

    let input = "4! 1! !jmp 2! 3!";
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![1, 3]);
}

#[test]
fn test_evaluator_jmpif() {
    let input = "1! 0! 5! jmpif! 2! 3!";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![1, 2, 3]);

    let input = "1! 1! 5! jmpif! 2! 3!";
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![1, 3]);

    let input = "5! 0! 1! !jmpif 2! 3!";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![1, 2, 3]);

    let input = "5! 1! 1! !jmpif 2! 3!";
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![1, 3]);
}

#[test]
fn test_evaluator_labels() {
    let input = "1! 2! end: end!";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![1, 2, 2]);

    let input = "1! 2! end: !end";
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![2, 1, 2]);
}

#[test]
fn test_evaluator_exit() {
    let input = "1! 2! exit 3!";
    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
    assert_eq!(evaluator.data, vec![1, 2]);
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        panic!("Usage: deque-rust <filepath>");
    }

    let input = fs::read_to_string(&args[1]).expect("Usage: deque-rust <filepath>");

    // let input = "3! !5 !2 sub! !add";
    // println!("input: {input}");

    let tokenizer = Tokenizer::default();
    let tokens = tokenizer.parse(input.into());
    // println!("tokens: {tokens:?}");

    let mut evaluator = Evaluator::new();
    evaluator.evaluate(tokens);
}
