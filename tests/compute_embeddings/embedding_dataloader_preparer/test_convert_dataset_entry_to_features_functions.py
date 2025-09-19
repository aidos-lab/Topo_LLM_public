"""Tests functions for converting dataset entries to features in embedding_dataloader_preparer module."""

import copy
import logging

import pytest
from transformers.tokenization_utils_base import BatchEncoding

from topollm.compute_embeddings.embedding_dataloader_preparer.convert_dataset_entry_to_features_functions import (
    convert_dataset_entry_to_features,
    convert_dataset_entry_to_features_luster_data,
)
from topollm.compute_embeddings.embedding_dataloader_preparer.extract_spans import debug_str_masks
from topollm.config_classes.language_model.language_model_config import LanguageModelConfig
from topollm.config_classes.tokenizer.tokenizer_config import TokenizerConfig
from topollm.model_handling.tokenizer.load_tokenizer import load_modified_tokenizer
from topollm.typing.enums import LMmode, TaskType, Verbosity

# # # # # # # # # # # # # # # #
# START Global dataset entry examples for LUSTER dataset type.


# Small example with 4 entries.
dataset_entry_luster_data_example_small: dict[
    str,
    list,
] = {
    "dialogue_id": ["PMUL4488.json", "PMUL2599.json", "MUL2387.json", "MUL2693.json"],
    "turn_id": [0, 4, 2, 2],
    "source": [
        " user : i am looking for a place to dine. the restaurant should serve hungarian food and should be in the south.</s> emotion : neutral</s> domain : restaurant</s> state : food hungarian ; price range unknown ; name unknown ; area south ; book time unknown ; book day unknown ; book people unknown</s> database : no entity found in database</s> action :",  # noqa: E501
        "user : i'm looking for a 0 star hotel that is expensive.</s> system : i'm sorry there are no hotels that fit that criteria. would you like a different amount of stars?</s> user : what star ratings do you have for hotels in the centre?</s> system : there are 3 and 4 star hotels.</s> user : can you check for one in the moderate price range.</s> emotion : neutral</s> domain : hotel</s> state : name unknown ; area unknown ; parking unknown ; price range moderate ; stars dontcare ; internet unknown ; type unknown ; book stay unknown ; book day unknown ; book people unknown</s> database : 18 found in database - address sleeperz hotel, station road ; area centre ; internet yes ; parking no ; name cityroomz ; phone 01223304050 ; postcode cb12tz ; pricerange moderate ; stars 0 ; type hotel ; ref 29q1x35w</s> action :",  # noqa: E501
        "user : hi. i'm looking for a restaurant. i think it's called the rice ship or rice boat or something like that.</s> system : the rice boat is located in the west and is in the expensive range. would you like to book a reservation?</s> user : yes please. thanks for your help.</s> emotion : satisfied</s> domain : restaurant</s> state : food unknown ; price range unknown ; name rice boat ; area unknown ; book time unknown ; book day unknown ; book people unknown</s> database : 1 found in database - address 37 newnham road newnham ; area west ; food indian ; name rice boat ; phone 01223302800 ; postcode cb39ey ; pricerange expensive ; ref 24akjdgh</s> action :",  # noqa: E501
        "user : where is whipple museum of the history of science located? </s> system : the address is free school lane and the postcode is cb23rh. </s> user : what is the type of attraction and area for the whipple museum?</s> emotion : neutral</s> domain : attraction</s> state : type museum ; name whipple museum of the history of science ; area unknown</s> database : 1 found in database - address free school lane ; area centre ; entrance fee free ; name whipple museum of the history of science ; phone 01223330906 ; postcode cb23rh ; type museum</s> action :",  # noqa: E501
    ],
    "target": [
        " nooffer food south</s> conduct : apologetic</s> system : regretfully, we have nothing like that in the south.</s>",  # noqa: E501
        " inform price range moderate ; inform choice 18 ; inform type hotel</s> conduct : neutral</s> system : i have 18 different hotels in the moderate price range. is there a certain area you would like?</s>",  # noqa: E501
        " request book people</s> conduct : neutral</s> system : how many people would you like to reserve a table for?</s>",  # noqa: E501
        " inform type museum ; inform area centre ; inform name the whipple ; request more</s> conduct : neutral</s> system : the whipple is a museum type attraction located in the centre area. can i do anything else for you?</s>",  # noqa: E501
    ],
    "source_target": [
        " user : i am looking for a place to dine. the restaurant should serve hungarian food and should be in the south.</s> emotion : neutral</s> domain : restaurant</s> state : food hungarian ; price range unknown ; name unknown ; area south ; book time unknown ; book day unknown ; book people unknown</s> database : no entity found in database</s> action : nooffer food south</s> conduct : apologetic</s> system : regretfully, we have nothing like that in the south.</s>",  # noqa: E501
        "user : i'm looking for a 0 star hotel that is expensive.</s> system : i'm sorry there are no hotels that fit that criteria. would you like a different amount of stars?</s> user : what star ratings do you have for hotels in the centre?</s> system : there are 3 and 4 star hotels.</s> user : can you check for one in the moderate price range.</s> emotion : neutral</s> domain : hotel</s> state : name unknown ; area unknown ; parking unknown ; price range moderate ; stars dontcare ; internet unknown ; type unknown ; book stay unknown ; book day unknown ; book people unknown</s> database : 18 found in database - address sleeperz hotel, station road ; area centre ; internet yes ; parking no ; name cityroomz ; phone 01223304050 ; postcode cb12tz ; pricerange moderate ; stars 0 ; type hotel ; ref 29q1x35w</s> action : inform price range moderate ; inform choice 18 ; inform type hotel</s> conduct : neutral</s> system : i have 18 different hotels in the moderate price range. is there a certain area you would like?</s>",  # noqa: E501
        "user : hi. i'm looking for a restaurant. i think it's called the rice ship or rice boat or something like that.</s> system : the rice boat is located in the west and is in the expensive range. would you like to book a reservation?</s> user : yes please. thanks for your help.</s> emotion : satisfied</s> domain : restaurant</s> state : food unknown ; price range unknown ; name rice boat ; area unknown ; book time unknown ; book day unknown ; book people unknown</s> database : 1 found in database - address 37 newnham road newnham ; area west ; food indian ; name rice boat ; phone 01223302800 ; postcode cb39ey ; pricerange expensive ; ref 24akjdgh</s> action : request book people</s> conduct : neutral</s> system : how many people would you like to reserve a table for?</s>",  # noqa: E501
        "user : where is whipple museum of the history of science located? </s> system : the address is free school lane and the postcode is cb23rh. </s> user : what is the type of attraction and area for the whipple museum?</s> emotion : neutral</s> domain : attraction</s> state : type museum ; name whipple museum of the history of science ; area unknown</s> database : 1 found in database - address free school lane ; area centre ; entrance fee free ; name whipple museum of the history of science ; phone 01223330906 ; postcode cb23rh ; type museum</s> action : inform type museum ; inform area centre ; inform name the whipple ; request more</s> conduct : neutral</s> system : the whipple is a museum type attraction located in the centre area. can i do anything else for you?</s>",  # noqa: E501
    ],
}

# Medium example with 16 entries.
dataset_entry_luster_data_example_medium: dict[
    str,
    list,
] = {
    "dialogue_id": [
        "PMUL0207.json",
        "SNG0457.json",
        "SNG0845.json",
        "PMUL3797.json",
        "PMUL3979.json",
        "PMUL3630.json",
        "MUL0848.json",
        "PMUL2255.json",
        "PMUL3434.json",
        "SNG01586.json",
        "PMUL2324.json",
        "SNG0314.json",
        "MUL2456.json",
        "SNG01699.json",
        "PMUL4793.json",
        "MUL0065.json",
    ],
    "turn_id": [0, 4, 0, 10, 6, 6, 2, 4, 2, 8, 10, 6, 0, 2, 10, 12],
    "source": [
        " user : i want to eat some greek food. </s> emotion : neutral</s> domain : restaurant</s> state : food greek ; price range unknown ; name unknown ; area unknown ; book time unknown ; book day unknown ; book people unknown</s> database : no entity found in database</s> action :",  # noqa: E501
        "user : hi! are there any expensive greek restaurants in town?</s> system : unfortunately there are none. would you like to try something else?</s> user : are there any expensive french restaurants in town?</s> system : yes, there are two: restaurant two two in the north at 22 chesterton road chesterton cb43ax and cote in the centre at bridge street city centre cb21uf. anything else?</s> user : can i get a table at 16:30 instead?</s> emotion : neutral</s> domain : restaurant</s> state : food french ; price range expensive ; name cote ; area unknown ; book time 16:30 ; book day sunday ; book people 1</s> database : 1 found in database - address bridge street city centre ; area centre ; food french ; name cote ; phone 01223311053 ; postcode cb21uf ; pricerange expensive ; ref 48aqq83y</s> action :",  # noqa: E501
        " user : i am looking for a hotel that is expensive and has free parking. </s> emotion : neutral</s> domain : hotel</s> state : name unknown ; area unknown ; parking yes ; price range expensive ; stars unknown ; internet unknown ; type hotel ; book stay unknown ; book day unknown ; book people unknown</s> database : 5 found in database - address 15-17 norman way, coldhams business park ; area east ; internet yes ; parking yes ; name express by holiday inn cambridge ; phone 01223866800 ; postcode cb13lh ; pricerange expensive ; stars 2 ; type hotel ; ref bohpjife</s> action :",  # noqa: E501
        "user : yes, i am looking for a molecular gastronomy restaurant in the centre.</s> system : i am sorry. there are not matches. would you like to try a different type of restaurant?</s> user : do you have any molecular gastronomy restaurants at all?</s> system : i'm afraid not, none in cambridge. would you like a different cuisine?</s> user : how about an expensive thai restaurant?</s> system : there is one in the west part of town, and one in the centre part of town. which area do you prefer?</s> user : i would prefer one in the centre part of town, do they have openings for tonight?</s> system : bangkok city is a thai restaurant that would meet your needs.</s> user : great! can i have the address and phone number please?</s> system : yes, the address is 24 green street city centre and phone number is 01223354382.</s> user : can you help me with attractions? i'd like a place to go in the same part of town as the restaurant.</s> emotion : neutral</s> domain : attraction</s> state : type unknown ; name unknown ; area centre</s> database : 44 found in database - address 1 station road ; area centre ; entrance fee 5 pounds ; name club salsa ; phone 07782218745 ; postcode cb12jb ; type nightclub</s> action :",  # noqa: E501
        "user : are there any accommodations in the east part of town that off free parking?</s> system : i have 6 places. would you prefer a guesthouse or a hotel?</s> user : i want it to be a hotel and also include free wifi.</s> system : i have the express by holiday inn cambridge. would you like me to book it for you?</s> user : what is their star rating?</s> system : it has a two star rating</s> user : can you also tell me the address?</s> emotion : neutral</s> domain : hotel</s> state : name express by holiday inn cambridge ; area east ; parking yes ; price range unknown ; stars unknown ; internet yes ; type hotel ; book stay unknown ; book day unknown ; book people unknown</s> database : 1 found in database - address 15-17 norman way, coldhams business park ; area east ; internet yes ; parking yes ; name express by holiday inn cambridge ; phone 01223866800 ; postcode cb13lh ; pricerange expensive ; stars 2 ; type hotel ; ref cizys_mz</s> action :",  # noqa: E501
        "user : i am on a budget and looking for a cheap place to stay with free wifi.</s> system : i can help you with that! is there a specific area you prefer to stay in?</s> user : i'm not sure on the area of town. i do know i want a guesthouse. is there anything for me?</s> system : there are nine options that meet your criteria. would you like a recommendation or do you want to narrow it down further?</s> user : book for one and 2 nights starting friday one of your choice</s> system : will you have other guests with you?</s> user : just for one person please.</s> emotion : neutral</s> domain : hotel</s> state : name unknown ; area unknown ; parking unknown ; price range cheap ; stars unknown ; internet yes ; type guesthouse ; book stay 2 ; book day friday ; book people 1</s> database : 9 found in database - address 517a coldham lane ; area east ; internet yes ; parking yes ; name allenbell ; phone 01223210353 ; postcode cb13js ; pricerange cheap ; stars 4 ; type guesthouse ; ref by6e1mtc</s> action :",  # noqa: E501
        "user : i'm traveling to cambridge and looking for things to do in the town centre.</s> system : what kind of attraction? we've got architecture, museums, theatres, colleges, and clubs. do any of these categories appeal to you?</s> user : i'd like to check out a theatre please.</s> emotion : neutral</s> domain : attraction</s> state : type unknown ; name unknown ; area centre</s> database : 44 found in database - address anglia ruskin enterprise, east road ; area centre ; entrance fee ? ; name mumford theatre ; phone 08451962320 ; postcode cb11pt ; type theatre</s> action :",  # noqa: E501
        "user : i am seeking a concerthall in the west part of town. </s> system : there are none available at this time.</s> user : ok. could you try for a museum. </s> system : there are 7 museums in the west. cambridge and country folk museum charges 3.50 pounds for entry and the rest are free. would you like the address?</s> user : please provide me with entrance fee and postcode</s> emotion : neutral</s> domain : attraction</s> state : type museum ; name cambridge and county folk museum ; area west</s> database : 1 found in database - address 2-3 castle street ; area west ; entrance fee 3.50 pounds ; name cambridge and county folk museum ; phone 01223355159 ; postcode cb30aq ; type museum</s> action :",  # noqa: E501
        "user : i need a place to stay in the centre of cambridge that's very expensive</s> system : university arms hotel is located in the centre, is expensive and has 4 stars.</s> user : i need a 3 star hotel</s> emotion : neutral</s> domain : hotel</s> state : name unknown ; area centre ; parking unknown ; price range expensive ; stars 3 ; internet unknown ; type unknown ; book stay unknown ; book day unknown ; book people unknown</s> database : 1 found in database - address gonville place ; area centre ; internet yes ; parking yes ; name gonville hotel ; phone 01223366611 ; postcode cb11ly ; pricerange expensive ; stars 3 ; type hotel ; ref cb34k46t</s> action :",  # noqa: E501
        "user : i need to book a taxi departing from royal standard.</s> system : i can help with that, what is your destination?</s> user : i'm heading to the university arms hotel.</s> system : got it. and can i have a time please?</s> user : yes. i must arrive there by 15:45</s> system : have you in a gray skoda, 07138317821 is the contact info. </s> user : sounds good, thanks for the help!</s> system : is there anything else i can help you with?</s> user : that was everything, thanks!</s> emotion : satisfied</s> domain : taxi</s> state : leave at unknown ; destination university arms hotel ; departure royal standard ; arrive by 15:45</s> database : 1 found in database - type gray skoda ; phone 07138317821</s> action :",  # noqa: E501
        "user : hi, i am planning a trip and could use some help finding a restaurant.</s> system : to help narrow the search, can you tell me the price range you're looking for and what kind of food you want? </s> user : i would really love to try some moroccan food. and something expensive, too. </s> system : sorry we got no restaurant matching your request. can you make any changes?</s> user : okay, how about an expensive british place instead? still would prefer to be in the city centre, though.</s> system : i have three expensive british places in centre. how about fitzbillies? </s> user : yes, that sounds fine. can you give the phone number and address for fitzbillies? </s> system : the address is 51 trumpington street city centre and the phone number is 01223352500.</s> user : i am also looking for something fun to do in the centre but not sure what. can you make a recommendation?</s> system : there is scott polar museum, would you be interested in that? </s> user : yes that would be fine. i need the postcode, entrance fee and attraction type.</s> emotion : neutral</s> domain : attraction</s> state : type unknown ; name scott polar museum ; area centre</s> database : 1 found in database - address lensfield road ; area centre ; entrance fee free ; name scott polar museum ; phone 01223336540 ; postcode cb21er ; type museum</s> action :",  # noqa: E501
        "user : i'd like to find a train tuesday departing from leicester.</s> system : i have 19 trains departing on tuesday from leicester, i can narrow it down a bit if you have a preferred time to leave at or arrive by.</s> user : i would like to leave after 11:15. </s> system : i have many available. is your destination cambridge or do you have another location in mind?</s> user : yes i would like to go from leicester to cambridge and leave after 11:15 please. </s> system : the first available train leaving after 11:15 would be train tr4354 which departs at 12:09 and arrives at 13:54. would you like me to book you a ticket for this train?</s> user : yes i'll need 4 tickets please.</s> emotion : neutral</s> domain : train</s> state : leave at 11:15 ; destination cambridge ; day tuesday ; arrive by unknown ; departure leicester ; book people 4</s> database : 12 found in database - arriveby 13:54 ; day tuesday ; departure leicester ; destination cambridge ; duration 105 minutes ; leaveat 12:09 ; price 37.80 pounds ; trainid tr4354 ; ref plynp1vg</s> action :",  # noqa: E501
        " user : hi, i am looking for attractions in the centre.</s> emotion : neutral</s> domain : attraction</s> state : type unknown ; name unknown ; area centre</s> database : 44 found in database - address jesus lane ; area centre ; entrance fee free ; name all saints church ; phone 01223452587 ; postcode cb58bs ; type architecture</s> action :",  # noqa: E501
        "user : i'd like to find a place that i can get some northern european food.</s> system : unfortunately, i was unable to find a restaurant that serves northern european food. is there anything else i can help you with?</s> user : okay, what about a restaurant that serves modern european food?</s> emotion : neutral</s> domain : restaurant</s> state : food modern european ; price range unknown ; name unknown ; area unknown ; book time unknown ; book day unknown ; book people unknown</s> database : 5 found in database - address 83 regent street ; area centre ; food modern european ; name de luca cucina and bar ; phone 01223356666 ; postcode cb21aw ; pricerange moderate ; signature roasted barbary duck breast served with sweet potato wedges and mange tout with a red wine sauce ; ref cizys_mz</s> action :",  # noqa: E501
        "user : i would like to take a train from cambridge to peterbourough.</s> system : there are lots of those! what day are you traveling?</s> user : i am traveling on wednesday and want to leave sometime after 11:30.</s> system : train 1097 departs cambridge at 11:34 and arrives in peterborough at 12:24. how does that sound?</s> user : sounds great. i need it booked for 7 people. </s> system : booking was successful, the total fee is 115.5 gbp payable at the station . reference number is : f7pldbgu</s> user : thank you! i'm also looking for entertainment in the centre.</s> system : i am sorry there are no entertainment option in the center.</s> user : how about a nightclub then?</s> system : there are five nightclubs in the centre of town. i would highly recommend kambar as it's my favorite! entrance fee is 5 pounds. anything else i can do for you?</s> user : yes, thanks! what is the address and phone number for kambar?</s> emotion : satisfied</s> domain : attraction</s> state : type nightclub ; name unknown ; area centre</s> database : 5 found in database - address 1 wheeler street ; area centre ; entrance fee 5 pounds ; name kambar ; phone 01223842725 ; postcode cb23qb ; type nightclub</s> action :",  # noqa: E501
        "user : i'm looking for a restaurant, in the moderate price range, in the southern part of town.</s> system : pizza hut cherry hinton and restaurant alimentum both meet your criteria. would you like a table at one of these?</s> user : how about restaurant alimentum for 8 people on wednesday at 16:30? that would be good.</s> system : your booking was successful. reference number 6nol39q9. can i help you with anything else?</s> user : yes i also need a place to stay. i would prefer a guesthouse. that's the only thing that really matters.</s> system : there are 24 guesthouses. would you like to narrow this down by price range, area, star rating, or amenities?</s> user : i specifically want free parking and don't mind if there is no internet.</s> system : that narrows it down to 21 places. is there a price range, area, or star rating you're looking for?</s> user : how about a place with free parking and free wifi? </s> system : certainly, acorn guest house, alexander bed and breakfast, allenbell, and archway house are all four star rated.</s> user : great i need 8 people for 3 nights starting wednesday.</s> system : ok, i've booked you at the allenbell guesthouse 517a coldham lane for three nights starting wednesday for 8 guests. your reference number is 1e099y11. is there anything else?</s> user : is this acceptable? is there anything else i can help you with?</s> emotion : neutral</s> domain : hotel</s> state : name unknown ; area unknown ; parking yes ; price range unknown ; stars unknown ; internet yes ; type guesthouse ; book stay 3 ; book day wednesday ; book people 8</s> database : 21 found in database - address 52 gilbert road ; area north ; internet yes ; parking yes ; name archway house ; phone 01223575314 ; postcode cb43pe ; pricerange moderate ; stars 4 ; type guesthouse ; ref 1e099y11</s> action :",  # noqa: E501
    ],
    "target": [
        " request food ; nooffer</s> conduct : neutral</s> system : there is no match for that criterion, do you have any other food preferences?</s>",  # noqa: E501
        " inform reference 48aqq83y ; book</s> conduct : neutral</s> system : booked! the table will be reserved for 15 minutes. reference number is 48aqq83y.</s>",  # noqa: E501
        " inform choice 5 ; request area</s> conduct : neutral</s> system : i have about 5 great options for you. do you prefer a certain area in the city? </s>",  # noqa: E501
        " inform area centre area</s> conduct : neutral</s> system : we have 44 attraction in the centre area, and specific type in mind?</s>",  # noqa: E501
        " inform address 15 - 17 norman way ; inform address coldhams business park</s> conduct : neutral</s> system : it is located at 15-17 norman way, coldhams business park. </s>",  # noqa: E501
        " offerbook reference by6e1mtc ; offerbook name allenbell</s> conduct : neutral</s> system : booking was successful at the allenbell. the reference number is : by6e1mtc.</s>",  # noqa: E501
        " recommend address anglia ruskin enterprise ; recommend address east road ; recommend name mumford theatre</s> conduct : enthusiastic</s> system : i suggest the mumford theatre at anglia ruskin enterprise, east road.</s>",  # noqa: E501
        " inform name cambridge and county folk museum ; inform entrance fee 3.50 pounds ; inform postcode cb30aq</s> conduct : neutral</s> system : cambridge and county folk museum is 3.50 pounds for the entrance fee and the postcode is cb30aq.</s>",  # noqa: E501
        " inform name gonville hotel</s> conduct : enthusiastic</s> system : what about gonville hotel? it's lovely</s>",  # noqa: E501
        " bye</s> conduct : appreciative</s> system : thank you, have a great day.</s>",
        " inform type museum ; inform postcode cb21er ; inform entrance fee free entry ; inform name the scott polar</s> conduct : neutral</s> system : the scott polar is a museum in the cb21er post code. it has free entry.</s>",  # noqa: E501
        " inform reference plynp1vg ; request more ; book</s> conduct : neutral</s> system : perfect. i was able to book for tickets on that train and your reference number for the journey is plynp1vg. anything else i can help you with?</s>",  # noqa: E501
        " recommend type architecture ; inform area centre ; recommend entrance fee free ; recommend address jesus lane ; recommend name all saints church ; inform choice many ; request more</s> conduct : enthusiastic</s> system : fantastic! there are many attractions in the centre of town. i'd suggest checking out the architecture at all saints church on jesus lane. it has free entrance. would that work?</s>",  # noqa: E501
        " inform choice several ; request price range ; request area</s> conduct : neutral</s> system : i have several, can you be mores specific on price and area please?</s>",  # noqa: E501
        " inform address 1 wheeler street ; inform phone 01223842725 ; request more</s> conduct : neutral</s> system : address is 1 wheeler street. phone is 01223842725. what else can i do for you?</s>",  # noqa: E501
        " request more</s> conduct : neutral</s> system : will you be needing further assistance today?</s>",
    ],
    "source_target": [
        " user : i want to eat some greek food. </s> emotion : neutral</s> domain : restaurant</s> state : food greek ; price range unknown ; name unknown ; area unknown ; book time unknown ; book day unknown ; book people unknown</s> database : no entity found in database</s> action : request food ; nooffer</s> conduct : neutral</s> system : there is no match for that criterion, do you have any other food preferences?</s>",  # noqa: E501
        "user : hi! are there any expensive greek restaurants in town?</s> system : unfortunately there are none. would you like to try something else?</s> user : are there any expensive french restaurants in town?</s> system : yes, there are two: restaurant two two in the north at 22 chesterton road chesterton cb43ax and cote in the centre at bridge street city centre cb21uf. anything else?</s> user : can i get a table at 16:30 instead?</s> emotion : neutral</s> domain : restaurant</s> state : food french ; price range expensive ; name cote ; area unknown ; book time 16:30 ; book day sunday ; book people 1</s> database : 1 found in database - address bridge street city centre ; area centre ; food french ; name cote ; phone 01223311053 ; postcode cb21uf ; pricerange expensive ; ref 48aqq83y</s> action : inform reference 48aqq83y ; book</s> conduct : neutral</s> system : booked! the table will be reserved for 15 minutes. reference number is 48aqq83y.</s>",  # noqa: E501
        " user : i am looking for a hotel that is expensive and has free parking. </s> emotion : neutral</s> domain : hotel</s> state : name unknown ; area unknown ; parking yes ; price range expensive ; stars unknown ; internet unknown ; type hotel ; book stay unknown ; book day unknown ; book people unknown</s> database : 5 found in database - address 15-17 norman way, coldhams business park ; area east ; internet yes ; parking yes ; name express by holiday inn cambridge ; phone 01223866800 ; postcode cb13lh ; pricerange expensive ; stars 2 ; type hotel ; ref bohpjife</s> action : inform choice 5 ; request area</s> conduct : neutral</s> system : i have about 5 great options for you. do you prefer a certain area in the city? </s>",  # noqa: E501
        "user : yes, i am looking for a molecular gastronomy restaurant in the centre.</s> system : i am sorry. there are not matches. would you like to try a different type of restaurant?</s> user : do you have any molecular gastronomy restaurants at all?</s> system : i'm afraid not, none in cambridge. would you like a different cuisine?</s> user : how about an expensive thai restaurant?</s> system : there is one in the west part of town, and one in the centre part of town. which area do you prefer?</s> user : i would prefer one in the centre part of town, do they have openings for tonight?</s> system : bangkok city is a thai restaurant that would meet your needs.</s> user : great! can i have the address and phone number please?</s> system : yes, the address is 24 green street city centre and phone number is 01223354382.</s> user : can you help me with attractions? i'd like a place to go in the same part of town as the restaurant.</s> emotion : neutral</s> domain : attraction</s> state : type unknown ; name unknown ; area centre</s> database : 44 found in database - address 1 station road ; area centre ; entrance fee 5 pounds ; name club salsa ; phone 07782218745 ; postcode cb12jb ; type nightclub</s> action : inform area centre area</s> conduct : neutral</s> system : we have 44 attraction in the centre area, and specific type in mind?</s>",  # noqa: E501
        "user : are there any accommodations in the east part of town that off free parking?</s> system : i have 6 places. would you prefer a guesthouse or a hotel?</s> user : i want it to be a hotel and also include free wifi.</s> system : i have the express by holiday inn cambridge. would you like me to book it for you?</s> user : what is their star rating?</s> system : it has a two star rating</s> user : can you also tell me the address?</s> emotion : neutral</s> domain : hotel</s> state : name express by holiday inn cambridge ; area east ; parking yes ; price range unknown ; stars unknown ; internet yes ; type hotel ; book stay unknown ; book day unknown ; book people unknown</s> database : 1 found in database - address 15-17 norman way, coldhams business park ; area east ; internet yes ; parking yes ; name express by holiday inn cambridge ; phone 01223866800 ; postcode cb13lh ; pricerange expensive ; stars 2 ; type hotel ; ref cizys_mz</s> action : inform address 15 - 17 norman way ; inform address coldhams business park</s> conduct : neutral</s> system : it is located at 15-17 norman way, coldhams business park. </s>",  # noqa: E501
        "user : i am on a budget and looking for a cheap place to stay with free wifi.</s> system : i can help you with that! is there a specific area you prefer to stay in?</s> user : i'm not sure on the area of town. i do know i want a guesthouse. is there anything for me?</s> system : there are nine options that meet your criteria. would you like a recommendation or do you want to narrow it down further?</s> user : book for one and 2 nights starting friday one of your choice</s> system : will you have other guests with you?</s> user : just for one person please.</s> emotion : neutral</s> domain : hotel</s> state : name unknown ; area unknown ; parking unknown ; price range cheap ; stars unknown ; internet yes ; type guesthouse ; book stay 2 ; book day friday ; book people 1</s> database : 9 found in database - address 517a coldham lane ; area east ; internet yes ; parking yes ; name allenbell ; phone 01223210353 ; postcode cb13js ; pricerange cheap ; stars 4 ; type guesthouse ; ref by6e1mtc</s> action : offerbook reference by6e1mtc ; offerbook name allenbell</s> conduct : neutral</s> system : booking was successful at the allenbell. the reference number is : by6e1mtc.</s>",  # noqa: E501
        "user : i'm traveling to cambridge and looking for things to do in the town centre.</s> system : what kind of attraction? we've got architecture, museums, theatres, colleges, and clubs. do any of these categories appeal to you?</s> user : i'd like to check out a theatre please.</s> emotion : neutral</s> domain : attraction</s> state : type unknown ; name unknown ; area centre</s> database : 44 found in database - address anglia ruskin enterprise, east road ; area centre ; entrance fee ? ; name mumford theatre ; phone 08451962320 ; postcode cb11pt ; type theatre</s> action : recommend address anglia ruskin enterprise ; recommend address east road ; recommend name mumford theatre</s> conduct : enthusiastic</s> system : i suggest the mumford theatre at anglia ruskin enterprise, east road.</s>",  # noqa: E501
        "user : i am seeking a concerthall in the west part of town. </s> system : there are none available at this time.</s> user : ok. could you try for a museum. </s> system : there are 7 museums in the west. cambridge and country folk museum charges 3.50 pounds for entry and the rest are free. would you like the address?</s> user : please provide me with entrance fee and postcode</s> emotion : neutral</s> domain : attraction</s> state : type museum ; name cambridge and county folk museum ; area west</s> database : 1 found in database - address 2-3 castle street ; area west ; entrance fee 3.50 pounds ; name cambridge and county folk museum ; phone 01223355159 ; postcode cb30aq ; type museum</s> action : inform name cambridge and county folk museum ; inform entrance fee 3.50 pounds ; inform postcode cb30aq</s> conduct : neutral</s> system : cambridge and county folk museum is 3.50 pounds for the entrance fee and the postcode is cb30aq.</s>",  # noqa: E501
        "user : i need a place to stay in the centre of cambridge that's very expensive</s> system : university arms hotel is located in the centre, is expensive and has 4 stars.</s> user : i need a 3 star hotel</s> emotion : neutral</s> domain : hotel</s> state : name unknown ; area centre ; parking unknown ; price range expensive ; stars 3 ; internet unknown ; type unknown ; book stay unknown ; book day unknown ; book people unknown</s> database : 1 found in database - address gonville place ; area centre ; internet yes ; parking yes ; name gonville hotel ; phone 01223366611 ; postcode cb11ly ; pricerange expensive ; stars 3 ; type hotel ; ref cb34k46t</s> action : inform name gonville hotel</s> conduct : enthusiastic</s> system : what about gonville hotel? it's lovely</s>",  # noqa: E501
        "user : i need to book a taxi departing from royal standard.</s> system : i can help with that, what is your destination?</s> user : i'm heading to the university arms hotel.</s> system : got it. and can i have a time please?</s> user : yes. i must arrive there by 15:45</s> system : have you in a gray skoda, 07138317821 is the contact info. </s> user : sounds good, thanks for the help!</s> system : is there anything else i can help you with?</s> user : that was everything, thanks!</s> emotion : satisfied</s> domain : taxi</s> state : leave at unknown ; destination university arms hotel ; departure royal standard ; arrive by 15:45</s> database : 1 found in database - type gray skoda ; phone 07138317821</s> action : bye</s> conduct : appreciative</s> system : thank you, have a great day.</s>",  # noqa: E501
        "user : hi, i am planning a trip and could use some help finding a restaurant.</s> system : to help narrow the search, can you tell me the price range you're looking for and what kind of food you want? </s> user : i would really love to try some moroccan food. and something expensive, too. </s> system : sorry we got no restaurant matching your request. can you make any changes?</s> user : okay, how about an expensive british place instead? still would prefer to be in the city centre, though.</s> system : i have three expensive british places in centre. how about fitzbillies? </s> user : yes, that sounds fine. can you give the phone number and address for fitzbillies? </s> system : the address is 51 trumpington street city centre and the phone number is 01223352500.</s> user : i am also looking for something fun to do in the centre but not sure what. can you make a recommendation?</s> system : there is scott polar museum, would you be interested in that? </s> user : yes that would be fine. i need the postcode, entrance fee and attraction type.</s> emotion : neutral</s> domain : attraction</s> state : type unknown ; name scott polar museum ; area centre</s> database : 1 found in database - address lensfield road ; area centre ; entrance fee free ; name scott polar museum ; phone 01223336540 ; postcode cb21er ; type museum</s> action : inform type museum ; inform postcode cb21er ; inform entrance fee free entry ; inform name the scott polar</s> conduct : neutral</s> system : the scott polar is a museum in the cb21er post code. it has free entry.</s>",  # noqa: E501
        "user : i'd like to find a train tuesday departing from leicester.</s> system : i have 19 trains departing on tuesday from leicester, i can narrow it down a bit if you have a preferred time to leave at or arrive by.</s> user : i would like to leave after 11:15. </s> system : i have many available. is your destination cambridge or do you have another location in mind?</s> user : yes i would like to go from leicester to cambridge and leave after 11:15 please. </s> system : the first available train leaving after 11:15 would be train tr4354 which departs at 12:09 and arrives at 13:54. would you like me to book you a ticket for this train?</s> user : yes i'll need 4 tickets please.</s> emotion : neutral</s> domain : train</s> state : leave at 11:15 ; destination cambridge ; day tuesday ; arrive by unknown ; departure leicester ; book people 4</s> database : 12 found in database - arriveby 13:54 ; day tuesday ; departure leicester ; destination cambridge ; duration 105 minutes ; leaveat 12:09 ; price 37.80 pounds ; trainid tr4354 ; ref plynp1vg</s> action : inform reference plynp1vg ; request more ; book</s> conduct : neutral</s> system : perfect. i was able to book for tickets on that train and your reference number for the journey is plynp1vg. anything else i can help you with?</s>",  # noqa: E501
        " user : hi, i am looking for attractions in the centre.</s> emotion : neutral</s> domain : attraction</s> state : type unknown ; name unknown ; area centre</s> database : 44 found in database - address jesus lane ; area centre ; entrance fee free ; name all saints church ; phone 01223452587 ; postcode cb58bs ; type architecture</s> action : recommend type architecture ; inform area centre ; recommend entrance fee free ; recommend address jesus lane ; recommend name all saints church ; inform choice many ; request more</s> conduct : enthusiastic</s> system : fantastic! there are many attractions in the centre of town. i'd suggest checking out the architecture at all saints church on jesus lane. it has free entrance. would that work?</s>",  # noqa: E501
        "user : i'd like to find a place that i can get some northern european food.</s> system : unfortunately, i was unable to find a restaurant that serves northern european food. is there anything else i can help you with?</s> user : okay, what about a restaurant that serves modern european food?</s> emotion : neutral</s> domain : restaurant</s> state : food modern european ; price range unknown ; name unknown ; area unknown ; book time unknown ; book day unknown ; book people unknown</s> database : 5 found in database - address 83 regent street ; area centre ; food modern european ; name de luca cucina and bar ; phone 01223356666 ; postcode cb21aw ; pricerange moderate ; signature roasted barbary duck breast served with sweet potato wedges and mange tout with a red wine sauce ; ref cizys_mz</s> action : inform choice several ; request price range ; request area</s> conduct : neutral</s> system : i have several, can you be mores specific on price and area please?</s>",  # noqa: E501
        "user : i would like to take a train from cambridge to peterbourough.</s> system : there are lots of those! what day are you traveling?</s> user : i am traveling on wednesday and want to leave sometime after 11:30.</s> system : train 1097 departs cambridge at 11:34 and arrives in peterborough at 12:24. how does that sound?</s> user : sounds great. i need it booked for 7 people. </s> system : booking was successful, the total fee is 115.5 gbp payable at the station . reference number is : f7pldbgu</s> user : thank you! i'm also looking for entertainment in the centre.</s> system : i am sorry there are no entertainment option in the center.</s> user : how about a nightclub then?</s> system : there are five nightclubs in the centre of town. i would highly recommend kambar as it's my favorite! entrance fee is 5 pounds. anything else i can do for you?</s> user : yes, thanks! what is the address and phone number for kambar?</s> emotion : satisfied</s> domain : attraction</s> state : type nightclub ; name unknown ; area centre</s> database : 5 found in database - address 1 wheeler street ; area centre ; entrance fee 5 pounds ; name kambar ; phone 01223842725 ; postcode cb23qb ; type nightclub</s> action : inform address 1 wheeler street ; inform phone 01223842725 ; request more</s> conduct : neutral</s> system : address is 1 wheeler street. phone is 01223842725. what else can i do for you?</s>",  # noqa: E501
        "user : i'm looking for a restaurant, in the moderate price range, in the southern part of town.</s> system : pizza hut cherry hinton and restaurant alimentum both meet your criteria. would you like a table at one of these?</s> user : how about restaurant alimentum for 8 people on wednesday at 16:30? that would be good.</s> system : your booking was successful. reference number 6nol39q9. can i help you with anything else?</s> user : yes i also need a place to stay. i would prefer a guesthouse. that's the only thing that really matters.</s> system : there are 24 guesthouses. would you like to narrow this down by price range, area, star rating, or amenities?</s> user : i specifically want free parking and don't mind if there is no internet.</s> system : that narrows it down to 21 places. is there a price range, area, or star rating you're looking for?</s> user : how about a place with free parking and free wifi? </s> system : certainly, acorn guest house, alexander bed and breakfast, allenbell, and archway house are all four star rated.</s> user : great i need 8 people for 3 nights starting wednesday.</s> system : ok, i've booked you at the allenbell guesthouse 517a coldham lane for three nights starting wednesday for 8 guests. your reference number is 1e099y11. is there anything else?</s> user : is this acceptable? is there anything else i can help you with?</s> emotion : neutral</s> domain : hotel</s> state : name unknown ; area unknown ; parking yes ; price range unknown ; stars unknown ; internet yes ; type guesthouse ; book stay 3 ; book day wednesday ; book people 8</s> database : 21 found in database - address 52 gilbert road ; area north ; internet yes ; parking yes ; name archway house ; phone 01223575314 ; postcode cb43pe ; pricerange moderate ; stars 4 ; type guesthouse ; ref 1e099y11</s> action : request more</s> conduct : neutral</s> system : will you be needing further assistance today?</s>",  # noqa: E501
    ],
}

# END Global dataset entry examples for LUSTER dataset type.
# # # # # # # # # # # # # # # #


@pytest.fixture
def dataset_entry() -> dict[str, list]:
    """Fixture for dataset entry."""
    # Return a deep copy of the global dataset entry example to avoid mutation issues in tests.
    # TODO: Make the fixture parameterized to test multiple examples.
    # return copy.deepcopy(dataset_entry_luster_data_example_small)
    return copy.deepcopy(dataset_entry_luster_data_example_medium)


def check_features_basic_properties(
    features: BatchEncoding,
    dataset_entry: dict[str, list],
    column_name: str,
) -> None:
    """Check basic properties of the features."""
    assert isinstance(  # noqa: S101 - pytest assertion
        features,
        BatchEncoding,
    )
    assert "input_ids" in features  # noqa: S101 - pytest assertion
    assert "attention_mask" in features  # noqa: S101 - pytest assertion
    assert len(features.input_ids) == len(dataset_entry[column_name])  # noqa: S101 - pytest assertion


def test_convert_dataset_entry_to_features(
    dataset_entry: dict[str, list],
    language_model_config: LanguageModelConfig,
    tokenizer_config: TokenizerConfig,
    verbosity: Verbosity,
    logger_fixture: logging.Logger,
) -> None:
    """Test convert_dataset_entry_to_features function."""
    column_name = "source_target"

    (
        tokenizer,
        _tokenizer_modifier,
    ) = load_modified_tokenizer(
        language_model_config=language_model_config,
        tokenizer_config=tokenizer_config,
        verbosity=verbosity,
        logger=logger_fixture,
    )

    features: BatchEncoding = convert_dataset_entry_to_features(
        dataset_entry=dataset_entry,
        tokenizer=tokenizer,
        column_name=column_name,
        max_length=512,
    )

    check_features_basic_properties(
        features=features,
        dataset_entry=dataset_entry,
        column_name=column_name,
    )


def test_convert_dataset_entry_to_features_luster_data(
    dataset_entry: dict[str, list],
    tokenizer_config: TokenizerConfig,
    verbosity: Verbosity,
    logger_fixture: logging.Logger,
) -> None:
    """Test convert_dataset_entry_to_features_luster_data function."""
    column_name = "source_target"

    language_model_config = LanguageModelConfig(
        lm_mode=LMmode.CLM,
        task_type=TaskType.CAUSAL_LM,
        pretrained_model_name_or_path="microsoft/Phi-3.5-mini-instruct",
        short_model_name="Phi-3.5-mini-instruct",
    )

    (
        tokenizer,
        _tokenizer_modifier,
    ) = load_modified_tokenizer(
        language_model_config=language_model_config,
        tokenizer_config=tokenizer_config,
        verbosity=verbosity,
        logger=logger_fixture,
    )

    features: BatchEncoding = convert_dataset_entry_to_features_luster_data(
        dataset_entry=dataset_entry,
        tokenizer=tokenizer,
        column_name=column_name,
        max_length=512,
    )

    # Inspect one example visuall
    debug_str: str = debug_str_masks(
        features=features,
        tokenizer=tokenizer,
        mask_keys=None,
    )
    if verbosity >= Verbosity.NORMAL:
        logger_fixture.info(
            msg=f"debug_str:\n{debug_str}",  # noqa: G004 - low overhead
        )

    check_features_basic_properties(
        features=features,
        dataset_entry=dataset_entry,
        column_name=column_name,
    )
