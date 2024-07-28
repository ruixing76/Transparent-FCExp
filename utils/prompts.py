# prompts to generate explanation
gpt_prompt_core = (
    "Instructions: You are required to write an accurate, coherent and logically consistent explanation for "
    "the claim based on the given veracity and list of reasons in one paragraph. Use an unbiased and "
    "journalistic tone. When citing several search results, use [1][2][3]. Ensure to cite each reason exactly once. "
    "Additionally, if a sentence needs to cite a reason, include only one reason per sentence.\n"
    "Reasons:<reasons>Claim:<claim>\nVeracity:<veracity>\nExplanation:")

llama_prompt_core = (
    "<s>[INST] <<SYS>> Instructions: You are required to write an accurate, coherent and logically "
    "consistent explanation for the claim based on the given veracity and list of reasons in one "
    "paragraph. Use an unbiased and journalistic tone. When citing several search results, "
    "use [1][2][3]. Ensure to cite each reason exactly once. Additionally, if a sentence needs to "
    "cite a reason, include only one reason per sentence. <</SYS>> "
    "\nReasons:<reasons>Claim:<claim>\nVeracity:<veracity>\n [/INST]"
    "Explanation:\n")

in_context_example = (
    "Demonstration:\n\nReasons:\nReason [0]: Anglerfish may have a reputation for being among the "
    "creepier-looking ocean-dwellers, but it’s not because they grow to be seven feet long, "
    "as a viral image on Facebook claims.\nReason [1]: The Jan. 12 post shows a young girl reaching "
    "toward what appears to be a very large anglerfish mounted on display at a museum.\nReason [2]: "
    "The text above the image reads, \"So,... I’ve spent my entire life thinking the Deep Sea Angler "
    "Fish was about the size of a Nerf football. How horrifying is it that a typical specimen is "
    "SEVEN FEET LONG?\"\nReason [3]: What’s more, the picture referenced in the Facebook post alleging that "
    "anglerfish are typically 7 feet is taken from the Australian Museum’s 2012 exhibit titled  \"Deep "
    "Oceans\"\nReason [4]: The anglerfish in the photo is actually a large-scale sculpture model of the fish made of "
    "plaster.\nReason [5]: When the exhibit opened in June 2012, The Sydney Morning Herald reported "
    "on how the exhibit’s team had created an \"oversized anglerfish\" and listed the many steps in "
    "making it: \"Pieces such as the oversized anglerfish, with huge fangs and antenna-like flashing "
    "rod to attract prey, begin with cutting and welding a metal frame, then sculpting material over "
    "it and, finally, hand painting it,\" the story says.\n\nClaim:The typical anglerfish is seven "
    "feet long.\nVeracity:false\nExtracted Reasons:[3,5]\nJustification: Reason [3] establishes that the Facebook "
    "post's claim relies on a picture from the Australian Museum's 2012 exhibit. Reason [5] then reveals that the "
    "anglerfish in the exhibit is an oversized sculpture, not an actual specimen. Together, these reasons logically "
    "demonstrate that the viral claim of typical anglerfish being seven feet long is false, as it is based on a "
    "misrepresented image from an exhibit.")

# prompts to retrieve evidences
gpt_prompt_full = (
        "Instructions: "
        "You are required to retrieve a subset of reasons from the provided full reasons. The sentences in "
        "this subset should be coherent and logically consistent, presenting the most crucial information necessary to "
        "establish the veracity of the claim. Aim for the minimum number of sentences in the subset while maintaining "
        "the completeness and clarity. When extracting reasons, use [1,2,3]. At last, provide a justification "
        "explaining why they are good reasons and how they form a logically consistent reasoning process." + in_context_example +
        "Reasons:<reasons>Claim:<claim>\nVeracity:<veracity>\nExtracted Reasons:\n"
)

llama_prompt_full = (
        "<s>[INST] <<SYS>> Instructions: "
        "You are required to retrieve a subset of reasons from the provided full reasons. "
        "The sentences in this subset should be coherent and logically consistent, presenting the most crucial "
        "information necessary to establish the veracity of the claim. Aim for the minimum number of "
        "sentences in the subset while maintaining the completeness and clarity. When extracting reasons, "
        "use [1,2,3]. At last, provide a justification explaining why they are good reasons and how they form a "
        "logically consistent reasoning process. \n" + in_context_example + " <</SYS>>" +
        "Reasons:<reasons>Claim:<claim>\nVeracity:<veracity>\n [/INST]Extracted Reasons:\n"
)
