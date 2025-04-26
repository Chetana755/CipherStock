// Sample questions data
const sampleQuestions = [
    {
        id: "q1",
        title: "What are the best strategies for dividend investing in the current market?",
        content: "I'm looking for advice on building a dividend portfolio that can generate stable income while also providing some growth opportunity. Which sectors and specific companies would you recommend focusing on right now?",
        author: {
            name: "Alex Johnson",
            role: "Financial Analyst",
            avatar: "AJ"
        },
        timestamp: "2h ago",
        upvotes: 24,
        downvotes: 2,
        tags: ["Investing", "Dividends", "Stocks"],
        views: 328,
        followStatus: false,
        userVoteStatus: 0, // 0: no vote, 1: upvoted, -1: downvoted
        answers: [
            {
                id: "a1",
                author: {
                    name: "Sarah Lee",
                    role: "Investment Advisor",
                    avatar: "SL"
                },
                content: `For dividend investing in the current market, I recommend focusing on these sectors:
                <ul>
                    <li><strong>Utilities:</strong> Companies like NextEra Energy (NEE) that combine renewable energy growth with stable dividends</li>
                    <li><strong>Consumer Staples:</strong> Procter & Gamble (PG) and Coca-Cola (KO) have decades of dividend growth</li>
                    <li><strong>Healthcare:</strong> Johnson & Johnson (JNJ) and Pfizer (PFE) offer both value and yield</li>
                </ul>
                Consider a 60/40 split between dividend aristocrats and dividend growth stocks to balance immediate income with long-term appreciation.`,
                timestamp: "1h ago",
                upvotes: 12,
                downvotes: 0,
                userVoteStatus: 0
            },
            {
                id: "a2",
                author: {
                    name: "Michael Chen",
                    role: "Portfolio Manager",
                    avatar: "MC"
                },
                content: `I'd suggest a slightly different approach focusing on ETFs for diversification:
                <ul>
                    <li>SCHD (Schwab U.S. Dividend Equity ETF) - Low expense ratio with quality dividend growers</li>
                    <li>VYM (Vanguard High Dividend Yield ETF) - Broad exposure to high-yield dividend stocks</li>
                    <li>DGRO (iShares Core Dividend Growth ETF) - Focus on consistent dividend growth</li>
                </ul>
                This gives you sector diversification while maintaining focus on dividends. For individual stocks, look at companies with dividend payout ratios below 65% to ensure sustainability.`,
                timestamp: "45m ago",
                upvotes: 8,
                downvotes: 1,
                userVoteStatus: 0
            }
        ]
    },
    {
        id: "q2",
        title: "How will the latest Fed interest rate decision affect the bond market?",
        content: "With the Federal Reserve's recent shift in policy, I'm trying to understand the implications for my fixed income investments. Should I be adjusting my bond allocation or duration strategy in light of the latest rate decision?",
        author: {
            name: "Jamie Wilson",
            role: "Retail Investor",
            avatar: "JW"
        },
        timestamp: "5h ago",
        upvotes: 18,
        downvotes: 1,
        tags: ["Bonds", "Interest Rates", "Fed"],
        views: 215,
        followStatus: false,
        userVoteStatus: 0,
        answers: [
            {
                id: "a3",
                author: {
                    name: "David Goldstein",
                    role: "Fixed Income Analyst",
                    avatar: "DG"
                },
                content: `The Fed's recent decision has several implications for bond investors:
                
                1. <strong>Short-term bonds</strong> will see immediate price pressure as yields adjust to the new rate environment
                2. <strong>Intermediate bonds</strong> (3-7 years) may present the best value in the current yield curve
                3. <strong>TIPS</strong> (Treasury Inflation-Protected Securities) could be valuable if inflation remains above target
                
                Consider a barbell approach with some short-term holdings for liquidity and longer-duration bonds to lock in yields while they're still relatively high historically.`,
                timestamp: "3h ago",
                upvotes: 15,
                downvotes: 0,
                userVoteStatus: 0
            }
        ]
    },
    {
        id: "q3",
        title: "What's the optimal asset allocation for a 35-year-old with moderate risk tolerance?",
        content: "I'm 35, have an emergency fund, and am now focused on long-term investing. My retirement is about 30 years away. Given the current market conditions, what would be an appropriate asset allocation? I'd describe my risk tolerance as moderate - I can handle some volatility but don't want extreme risk.",
        author: {
            name: "Taylor Rodriguez",
            role: "Tech Professional",
            avatar: "TR"
        },
        timestamp: "1d ago",
        upvotes: 32,
        downvotes: 0,
        tags: ["Asset Allocation", "Retirement", "Risk Management"],
        views: 487,
        followStatus: false,
        userVoteStatus: 0,
        answers: [
            {
                id: "a4",
                author: {
                    name: "Patricia Hayden",
                    role: "Certified Financial Planner",
                    avatar: "PH"
                },
                content: `For a 35-year-old with moderate risk tolerance and a 30-year time horizon, I typically recommend:
                
                - <strong>70-75% Equities:</strong> Primarily low-cost index funds covering US, international developed, and emerging markets
                - <strong>20-25% Fixed Income:</strong> Mix of intermediate-term bonds and TIPS
                - <strong>5-10% Alternatives:</strong> REITs or commodity exposure for diversification
                
                The standard 60/40 portfolio is too conservative for your age and time horizon. With 30 years until retirement, you can afford to take more equity risk now and gradually reduce it as you approach retirement age.`,
                timestamp: "22h ago",
                upvotes: 26,
                downvotes: 1,
                userVoteStatus: 0
            },
            {
                id: "a5",
                author: {
                    name: "Robert Kim",
                    role: "Wealth Manager",
                    avatar: "RK"
                },
                content: `I'd suggest a slightly more detailed breakdown:
                
                <strong>Equities (70%):</strong>
                - 45% US Total Market (e.g., VTI)
                - 20% International Developed (e.g., VXUS)
                - 5% Emerging Markets (e.g., VWO)
                
                <strong>Fixed Income (25%):</strong>
                - 15% Intermediate-Term Treasuries
                - 10% Corporate Bonds
                
                <strong>Alternatives (5%):</strong>
                - REITs or Gold
                
                Just as important as your allocation is your behavior - stick to regular contributions regardless of market conditions and rebalance annually.`,
                timestamp: "18h ago",
                upvotes: 19,
                downvotes: 0,
                userVoteStatus: 0
            }
        ]
    }
];

// State
let questions = [...sampleQuestions];
let currentlyExpandedQuestion = null;
let activeFilter = 'all'; // Tracks current filter: 'all', 'answered', 'unanswered', 'votes', 'newest'
let userQuestions = []; // For "Your Questions" sidebar functionality
let userAnswers = []; // For "Your Answers" sidebar functionality
let userBookmarks = []; // For "Bookmarks" sidebar functionality

// DOM Elements - we'll initialize them in the initializeQA function
let questionsContainer;
let questionTemplate;
let answerTemplate;
let newQuestionContent;
let submitQuestionButton;
let questionTopicSelect;
let searchInput;
let sidebarMenuItems;
let filterBadges;

// Initialize the Q&A page
function initializeQA() {
    // Initialize DOM elements
    questionsContainer = document.getElementById('questionsContainer');
    questionTemplate = document.getElementById('question-template');
    answerTemplate = document.getElementById('answer-template');
    newQuestionContent = document.getElementById('newQuestionContent');
    submitQuestionButton = document.getElementById('submitQuestion');
    questionTopicSelect = document.getElementById('questionTopic');
    searchInput = document.querySelector('.search-bar input');
    sidebarMenuItems = document.querySelectorAll('.sidebar .menu-item');
    filterBadges = document.querySelectorAll('.filter-badge');
    
    // Initialize the page
    renderQuestions();
    setupEventListeners();
}

// Render questions based on filter and search criteria
function renderQuestions(searchTerm = '') {
    questionsContainer.innerHTML = '';
    
    // Get the active sidebar item
    const activeSidebar = document.querySelector('.sidebar .menu-item.active span').textContent;
    
    // Check if we need to show a special page instead of questions
    if (activeSidebar === 'Profile') {
        // Show Profile page
        const profilePage = document.createElement('div');
        profilePage.className = 'special-page glass';
        profilePage.innerHTML = `
            <div class="page-header">
                <i class="far fa-user"></i>
                <h2>Your Profile</h2>
            </div>
            <div class="profile-content">
                <div class="profile-avatar">YO</div>
                <div class="profile-info">
                    <h3>Financial Enthusiast</h3>
                    <p>Member since March 2025</p>
                    <div class="stats-row">
                        <div class="profile-stat">
                            <span class="count">${userQuestions.length}</span>
                            <span class="label">Questions</span>
                        </div>
                        <div class="profile-stat">
                            <span class="count">${userAnswers.length}</span>
                            <span class="label">Answers</span>
                        </div>
                        <div class="profile-stat">
                            <span class="count">${userBookmarks.length}</span>
                            <span class="label">Bookmarks</span>
                        </div>
                    </div>
                </div>
            </div>
            <div class="profile-actions">
                <button class="primary-button disabled">Edit Profile</button>
                <button class="secondary-button disabled">Notification Settings</button>
            </div>
        `;
        questionsContainer.appendChild(profilePage);
        return;
    } 
    else if (activeSidebar === 'Settings') {
        // Show Settings page
        const settingsPage = document.createElement('div');
        settingsPage.className = 'special-page glass';
        settingsPage.innerHTML = `
            <div class="page-header">
                <i class="fas fa-cog"></i>
                <h2>Settings</h2>
            </div>
            <div class="settings-content">
                <div class="settings-section">
                    <h3>Appearance</h3>
                    <div class="setting-row">
                        <span>Dark Mode</span>
                        <div class="toggle active"><div class="toggle-slider"></div></div>
                    </div>
                    <div class="setting-row">
                        <span>Compact View</span>
                        <div class="toggle"><div class="toggle-slider"></div></div>
                    </div>
                </div>
                <div class="settings-section">
                    <h3>Notifications</h3>
                    <div class="setting-row">
                        <span>Email Notifications</span>
                        <div class="toggle active"><div class="toggle-slider"></div></div>
                    </div>
                    <div class="setting-row">
                        <span>Question Answers</span>
                        <div class="toggle active"><div class="toggle-slider"></div></div>
                    </div>
                </div>
                <div class="settings-section">
                    <h3>Privacy</h3>
                    <div class="setting-row">
                        <span>Show Online Status</span>
                        <div class="toggle"><div class="toggle-slider"></div></div>
                    </div>
                </div>
            </div>
            <div class="profile-actions">
                <button class="primary-button disabled">Save Changes</button>
                <button class="secondary-button disabled">Reset to Defaults</button>
            </div>
        `;
        questionsContainer.appendChild(settingsPage);
        
        // Add event listeners to toggles
        settingsPage.querySelectorAll('.toggle').forEach(toggle => {
            toggle.addEventListener('click', function() {
                this.classList.toggle('active');
            });
        });
        
        return;
    }
    
    // If we get here, we're showing questions...
    
    // First filter by the active sidebar selection
    let sidebarFilteredQuestions = questions;
    
    // Sidebar filtering
    if (activeSidebar === 'Your Questions') {
        sidebarFilteredQuestions = questions.filter(q => userQuestions.includes(q.id) || q.author.name === 'You');
    } else if (activeSidebar === 'Your Answers') {
        sidebarFilteredQuestions = questions.filter(q => 
            userAnswers.includes(q.id) || q.answers.some(a => a.author.name === 'You')
        );
    } else if (activeSidebar === 'Bookmarks') {
        sidebarFilteredQuestions = questions.filter(q => userBookmarks.includes(q.id) || q.followStatus);
    }
    
    // Then apply the filter badges
    let filteredQuestions = sidebarFilteredQuestions;
    
    switch(activeFilter) {
        case 'answered':
            // Make sure we're only showing questions that have at least one answer
            filteredQuestions = sidebarFilteredQuestions.filter(q => q.answers && q.answers.length > 0);
            break;
        case 'unanswered':
            // Make sure we're only showing questions with no answers
            filteredQuestions = sidebarFilteredQuestions.filter(q => !q.answers || q.answers.length === 0);
            break;
        case 'votes':
            filteredQuestions = [...sidebarFilteredQuestions].sort((a, b) => 
                (b.upvotes - b.downvotes) - (a.upvotes - a.downvotes)
            );
            break;
        case 'newest':
            filteredQuestions = [...sidebarFilteredQuestions].sort((a, b) => {
                // This is simplified since we don't have actual timestamps
                // In a real app, you'd compare actual date objects
                if (a.timestamp.includes('now')) return -1;
                if (b.timestamp.includes('now')) return 1;
                if (a.timestamp.includes('ago') && b.timestamp.includes('ago')) {
                    const aTime = parseInt(a.timestamp);
                    const bTime = parseInt(b.timestamp);
                    if (a.timestamp.includes('m') && b.timestamp.includes('h')) return -1;
                    if (a.timestamp.includes('h') && b.timestamp.includes('m')) return 1;
                    if (a.timestamp.includes('h') && b.timestamp.includes('h')) return bTime - aTime;
                    if (a.timestamp.includes('m') && b.timestamp.includes('m')) return bTime - aTime;
                    if (a.timestamp.includes('d') && !b.timestamp.includes('d')) return 1;
                    if (!a.timestamp.includes('d') && b.timestamp.includes('d')) return -1;
                }
                return 0;
            });
            break;
    }
    
    // Finally, apply any search term filtering
    if (searchTerm) {
        filteredQuestions = filteredQuestions.filter(question => 
            question.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
            question.content.toLowerCase().includes(searchTerm.toLowerCase()) ||
            question.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase())) ||
            question.author.name.toLowerCase().includes(searchTerm.toLowerCase())
        );
    }
    
    if (filteredQuestions.length === 0) {
        const noResultsMessage = document.createElement('div');
        noResultsMessage.className = 'no-results glass';
        noResultsMessage.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-search"></i>
                <p>No results found for "${searchTerm}"</p>
                <button id="clearSearch" class="clear-search-btn">Clear Search</button>
            </div>
        `;
        questionsContainer.appendChild(noResultsMessage);
        
        // Add event listener to the clear search button
        const clearSearchBtn = document.getElementById('clearSearch');
        if (clearSearchBtn) {
            clearSearchBtn.addEventListener('click', () => {
                searchInput.value = '';
                renderQuestions();
            });
        }
    } else {
        filteredQuestions.forEach(question => {
            const questionElement = createQuestionElement(question);
            questionsContainer.appendChild(questionElement);
            
            // If this is the currently expanded question, show its answers
            if (currentlyExpandedQuestion === question.id) {
                const answersContainer = questionElement.querySelector('.answers-container');
                const answersList = answersContainer.querySelector('.answers-list');
                
                // Render answers
                question.answers.forEach(answer => {
                    const answerElement = createAnswerElement(answer);
                    answersList.appendChild(answerElement);
                });
                
                // Show answers container
                answersContainer.style.display = 'block';
            }
        });
    }
}

// Create a question element from the template
function createQuestionElement(question) {
    const questionElement = document.importNode(questionTemplate.content, true).querySelector('.question-card');
    
    // Set question data
    questionElement.dataset.questionId = question.id;
    
    // Author info
    const avatarSpan = questionElement.querySelector('.question-avatar span');
    avatarSpan.textContent = question.author.avatar;
    
    questionElement.querySelector('.author-name').textContent = question.author.name;
    questionElement.querySelector('.author-details').textContent = `${question.author.role} · ${question.timestamp}`;
    
    // Question content
    questionElement.querySelector('.question-title').textContent = question.title;
    
    const contentElement = questionElement.querySelector('.question-content');
    const searchTerm = searchInput.value.trim();
    if (searchTerm && (question.title.toLowerCase().includes(searchTerm.toLowerCase()) || 
                       question.content.toLowerCase().includes(searchTerm.toLowerCase()))) {
        contentElement.innerHTML = highlightSearchTerms(question.content, searchTerm);
    } else {
        contentElement.textContent = question.content;
    }
    
    // Tags
    const tagsContainer = questionElement.querySelector('.question-tags');
    tagsContainer.innerHTML = '';
    question.tags.forEach(tag => {
        const tagSpan = document.createElement('span');
        tagSpan.className = 'tag';
        tagSpan.textContent = tag;
        tagsContainer.appendChild(tagSpan);
    });
    
    // Stats
    questionElement.querySelector('.stat:nth-child(1) span').textContent = `${question.views} views`;
    questionElement.querySelector('.stat:nth-child(2) span').textContent = `${question.answers.length} ${question.answers.length === 1 ? 'answer' : 'answers'}`;
    
    // Vote count
    questionElement.querySelector('.vote-count').textContent = question.upvotes - question.downvotes;
    
    // Vote status
    const upvoteButton = questionElement.querySelector('.upvote');
    const downvoteButton = questionElement.querySelector('.downvote');
    
    if (question.userVoteStatus === 1) {
        upvoteButton.classList.add('active');
    } else if (question.userVoteStatus === -1) {
        downvoteButton.classList.add('active');
    }
    
    // Follow status
    const followButton = questionElement.querySelector('.follow-button');
    if (question.followStatus) {
        followButton.classList.add('active');
        followButton.innerHTML = '<i class="fas fa-bookmark"></i><span>Following</span>';
    }
    
    // Answer count 
    const answerCountElement = questionElement.querySelector('.answer-count h4');
    answerCountElement.textContent = `${question.answers.length} ${question.answers.length === 1 ? 'Answer' : 'Answers'}`;
    
    // Hide answers container initially
    const answersContainer = questionElement.querySelector('.answers-container');
    answersContainer.style.display = 'none';
    
    // Set up event listeners for this question
    setupQuestionEventListeners(questionElement, question);
    
    return questionElement;
}

// Create an answer element from the template
function createAnswerElement(answer) {
    const answerElement = document.importNode(answerTemplate.content, true).querySelector('.answer');
    
    // Set answer data
    answerElement.dataset.answerId = answer.id;
    
    // Author info
    const avatarSpan = answerElement.querySelector('.answer-avatar span');
    avatarSpan.textContent = answer.author.avatar;
    
    answerElement.querySelector('.author-name').textContent = answer.author.name;
    answerElement.querySelector('.author-details').textContent = `${answer.author.role} · ${answer.timestamp}`;
    
    // Answer content
    answerElement.querySelector('.answer-content').innerHTML = answer.content;
    
    // Vote count
    answerElement.querySelector('.vote-count').textContent = answer.upvotes - answer.downvotes;
    
    // Vote status
    const upvoteButton = answerElement.querySelector('.upvote');
    const downvoteButton = answerElement.querySelector('.downvote');
    
    if (answer.userVoteStatus === 1) {
        upvoteButton.classList.add('active');
    } else if (answer.userVoteStatus === -1) {
        downvoteButton.classList.add('active');
    }
    
    // Set up event listeners for this answer
    setupAnswerEventListeners(answerElement, answer);
    
    return answerElement;
}

// Set up event listeners for question elements
function setupQuestionEventListeners(questionElement, question) {
    const questionId = question.id;
    
    // Upvote button
    const upvoteButton = questionElement.querySelector('.upvote');
    upvoteButton.addEventListener('click', () => {
        handleVote(questionId, 'question', 'up');
    });
    
    // Downvote button
    const downvoteButton = questionElement.querySelector('.downvote');
    downvoteButton.addEventListener('click', () => {
        handleVote(questionId, 'question', 'down');
    });
    
    // Answer button
    const answerButton = questionElement.querySelector('.answer-button');
    answerButton.addEventListener('click', () => {
        toggleAnswerForm(questionElement, questionId);
    });
    
    // Follow button
    const followButton = questionElement.querySelector('.follow-button');
    followButton.addEventListener('click', () => {
        toggleFollowStatus(questionId, followButton);
    });
    
    // Show/hide answers
    questionElement.querySelector('.question-title').addEventListener('click', () => {
        toggleAnswersVisibility(questionElement, questionId);
    });
    
    // Add click event for answer count to toggle answers
    const answerStat = questionElement.querySelector('.stat:nth-child(2)');
    answerStat.addEventListener('click', () => {
        toggleAnswersVisibility(questionElement, questionId);
    });
    answerStat.style.cursor = 'pointer';
    
    // Submit answer
    const submitAnswerButton = questionElement.querySelector('.submit-answer');
    submitAnswerButton.addEventListener('click', () => {
        submitAnswer(questionElement, questionId);
    });
    
    // Cancel answer
    const cancelAnswerButton = questionElement.querySelector('.cancel-answer');
    cancelAnswerButton.addEventListener('click', () => {
        hideAnswerForm(questionElement);
    });
}

// Set up event listeners for answer elements
function setupAnswerEventListeners(answerElement, answer) {
    const answerId = answer.id;
    
    // Upvote button
    const upvoteButton = answerElement.querySelector('.upvote');
    upvoteButton.addEventListener('click', () => {
        handleVote(answerId, 'answer', 'up');
    });
    
    // Downvote button
    const downvoteButton = answerElement.querySelector('.downvote');
    downvoteButton.addEventListener('click', () => {
        handleVote(answerId, 'answer', 'down');
    });
}

// Toggle answer form visibility
function toggleAnswerForm(questionElement, questionId) {
    const answerForm = questionElement.querySelector('.answer-form');
    const answersContainer = questionElement.querySelector('.answers-container');
    
    // Make sure answers are visible first
    if (answersContainer.style.display === 'none') {
        toggleAnswersVisibility(questionElement, questionId);
    }
    
    // Toggle form visibility
    if (answerForm.classList.contains('hidden')) {
        answerForm.classList.remove('hidden');
        answerForm.querySelector('.answer-input').focus();
    } else {
        answerForm.classList.add('hidden');
    }
}

// Hide answer form
function hideAnswerForm(questionElement) {
    const answerForm = questionElement.querySelector('.answer-form');
    answerForm.classList.add('hidden');
    answerForm.querySelector('.answer-input').value = '';
}

// Toggle answers visibility
function toggleAnswersVisibility(questionElement, questionId) {
    const answersContainer = questionElement.querySelector('.answers-container');
    
    if (answersContainer.style.display === 'none') {
        // Show answers
        answersContainer.style.display = 'block';
        currentlyExpandedQuestion = questionId;
        
        // Populate answers if not already done
        const answersList = answersContainer.querySelector('.answers-list');
        if (answersList.children.length === 0) {
            const question = questions.find(q => q.id === questionId);
            if (question) {
                question.answers.forEach(answer => {
                    const answerElement = createAnswerElement(answer);
                    answersList.appendChild(answerElement);
                });
            }
        }
    } else {
        // Hide answers
        answersContainer.style.display = 'none';
        currentlyExpandedQuestion = null;
    }
}

// Handle vote actions (upvote/downvote)
function handleVote(id, type, direction) {
    let item;
    
    if (type === 'question') {
        item = questions.find(q => q.id === id);
    } else {
        // Find the answer in all questions
        for (const question of questions) {
            const answer = question.answers.find(a => a.id === id);
            if (answer) {
                item = answer;
                break;
            }
        }
    }
    
    if (!item) return;
    
    // Handle vote logic
    if (direction === 'up') {
        if (item.userVoteStatus === 1) {
            // Already upvoted, remove upvote
            item.upvotes--;
            item.userVoteStatus = 0;
        } else if (item.userVoteStatus === -1) {
            // Was downvoted, change to upvote
            item.downvotes--;
            item.upvotes++;
            item.userVoteStatus = 1;
        } else {
            // Not voted, add upvote
            item.upvotes++;
            item.userVoteStatus = 1;
        }
    } else if (direction === 'down') {
        if (item.userVoteStatus === -1) {
            // Already downvoted, remove downvote
            item.downvotes--;
            item.userVoteStatus = 0;
        } else if (item.userVoteStatus === 1) {
            // Was upvoted, change to downvote
            item.upvotes--;
            item.downvotes++;
            item.userVoteStatus = -1;
        } else {
            // Not voted, add downvote
            item.downvotes++;
            item.userVoteStatus = -1;
        }
    }
    
    // Re-render to update the UI
    renderQuestions(searchInput.value.trim());
}

// Toggle follow status
function toggleFollowStatus(questionId, button) {
    const question = questions.find(q => q.id === questionId);
    if (!question) return;
    
    question.followStatus = !question.followStatus;
    
    // Update the userBookmarks array for filtering
    if (question.followStatus) {
        // Add to bookmarks
        userBookmarks.push(questionId);
        button.classList.add('active');
        button.innerHTML = '<i class="fas fa-bookmark"></i><span>Following</span>';
        
        // Show a message to user if they want to switch to Bookmarks view
        const bookmarksMenuItem = Array.from(sidebarMenuItems).find(item => 
            item.querySelector('span').textContent === 'Bookmarks'
        );
        
        if (bookmarksMenuItem && !bookmarksMenuItem.classList.contains('active')) {
            // Optional: Add temporary visual indicator that this is now bookmarked
            button.style.transform = 'scale(1.2)';
            setTimeout(() => { button.style.transform = 'scale(1)'; }, 200);
        }
    } else {
        // Remove from bookmarks
        const index = userBookmarks.indexOf(questionId);
        if (index > -1) {
            userBookmarks.splice(index, 1);
        }
        button.classList.remove('active');
        button.innerHTML = '<i class="far fa-bookmark"></i><span>Follow</span>';
        
        // If we're in the Bookmarks view, we might need to re-render
        const activeSidebarItem = document.querySelector('.sidebar .menu-item.active span').textContent;
        if (activeSidebarItem === 'Bookmarks') {
            renderQuestions(searchInput.value.trim());
        }
    }
}

// Submit a new answer
function submitAnswer(questionElement, questionId) {
    const answerInput = questionElement.querySelector('.answer-input');
    const content = answerInput.value.trim();
    
    if (!content) return;
    
    const question = questions.find(q => q.id === questionId);
    if (!question) return;
    
    // Create new answer with HTML formatting preserved
    const newAnswer = {
        id: `a${Date.now()}`,
        author: {
            name: "You",
            role: "Financial Enthusiast",
            avatar: "YO"
        },
        content: content,
        timestamp: "Just now",
        upvotes: 0,
        downvotes: 0,
        userVoteStatus: 0
    };
    
    // Add to question's answers
    question.answers.push(newAnswer);
    
    // Also add to userAnswers array for filtering
    userAnswers.push(questionId);
    
    // Update the UI
    const answersList = questionElement.querySelector('.answers-list');
    const answerElement = createAnswerElement(newAnswer);
    answersList.appendChild(answerElement);
    
    // Update answer count in the question card
    const answerCountElement = questionElement.querySelector('.answer-count h4');
    answerCountElement.textContent = `${question.answers.length} ${question.answers.length === 1 ? 'Answer' : 'Answers'}`;
    
    // Update the stat counter for answers
    const answerStatElement = questionElement.querySelector('.stat:nth-child(2) span');
    answerStatElement.textContent = `${question.answers.length} ${question.answers.length === 1 ? 'answer' : 'answers'}`;
    
    // Clear and hide the form
    hideAnswerForm(questionElement);
}

// Handle posting a new question
function handleQuestionSubmit() {
    const content = newQuestionContent.value.trim();
    const topic = questionTopicSelect.value;
    
    if (!content) return;
    
    // Create a title from the first sentence or first X characters
    let title = content.split('.')[0];
    if (title.length > 100) {
        title = title.substring(0, 97) + '...';
    }
    
    // Create new question
    const newQuestion = {
        id: `q${Date.now()}`,
        title: title,
        content: content,
        author: {
            name: "You",
            role: "Financial Enthusiast",
            avatar: "YO"
        },
        timestamp: "Just now",
        upvotes: 0,
        downvotes: 0,
        tags: [topic.charAt(0).toUpperCase() + topic.slice(1)],
        views: 0,
        followStatus: false,
        userVoteStatus: 0,
        answers: []
    };
    
    // Add to questions and userQuestions arrays
    questions = [newQuestion, ...questions];
    userQuestions.push(newQuestion.id);
    
    // Switch to "Your Questions" view if not already there
    const yourQuestionsButton = Array.from(sidebarMenuItems).find(item => 
        item.querySelector('span').textContent === 'Your Questions'
    );
    
    if (yourQuestionsButton && !yourQuestionsButton.classList.contains('active')) {
        document.querySelector('.sidebar .menu-item.active').classList.remove('active');
        yourQuestionsButton.classList.add('active');
        
        // Reset filter badges to "All Questions"
        document.querySelector('.filter-badge.active').classList.remove('active');
        document.querySelector('.filter-badge:first-child').classList.add('active');
        activeFilter = 'all';
    }
    
    // Update the UI
    renderQuestions();
    
    // Clear the form
    newQuestionContent.value = '';
}

// Set up global event listeners
function setupEventListeners() {
    // Submit new question
    submitQuestionButton.addEventListener('click', handleQuestionSubmit);
    
    // Sidebar menu items
    sidebarMenuItems.forEach(item => {
        item.addEventListener('click', function() {
            // Remove active class from current active item
            document.querySelector('.sidebar .menu-item.active').classList.remove('active');
            // Add active class to clicked item
            this.classList.add('active');
            // Reset active filter to 'all' when switching sidebar items
            document.querySelector('.filter-badge.active').classList.remove('active');
            document.querySelector('.filter-badge:first-child').classList.add('active');
            activeFilter = 'all';
            // Render questions with new sidebar filter
            renderQuestions();
        });
    });
    
    // Filter badges (All Questions, Answered, Unanswered, Most Votes, Newest)
    document.querySelectorAll('.filter-badge').forEach(badge => {
        badge.addEventListener('click', function() {
            document.querySelector('.filter-badge.active').classList.remove('active');
            this.classList.add('active');
            
            // Set active filter based on badge text
            const filterText = this.textContent.toLowerCase();
            if (filterText.includes('all')) {
                activeFilter = 'all';
            } else if (filterText.includes('answered')) {
                activeFilter = 'answered';
            } else if (filterText.includes('unanswered')) {
                activeFilter = 'unanswered';
            } else if (filterText.includes('votes')) {
                activeFilter = 'votes';
            } else if (filterText.includes('newest')) {
                activeFilter = 'newest';
            }
            
            // Re-render with the new filter
            renderQuestions(searchInput.value.trim());
        });
    });
    
    // Topic items in sidebar
    document.querySelectorAll('.topic-item').forEach(topic => {
        topic.addEventListener('click', function() {
            const topicTag = this.querySelector('.topic-tag').textContent;
            searchInput.value = topicTag;
            renderQuestions(topicTag);
        });
    });
    
    // Search functionality
    searchInput.addEventListener('input', function(e) {
        const searchTerm = e.target.value.trim();
        renderQuestions(searchTerm);
    });
    
    // Search button click
    const searchButton = document.querySelector('.search-button');
    if (searchButton) {
        searchButton.addEventListener('click', function() {
            const searchTerm = searchInput.value.trim();
            renderQuestions(searchTerm);
        });
    }
    
    // Clear search when hitting escape key
    searchInput.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            searchInput.value = '';
            renderQuestions();
        }
        
        // Submit search on Enter key
        if (e.key === 'Enter') {
            const searchTerm = searchInput.value.trim();
            renderQuestions(searchTerm);
        }
    });
}

// Function to highlight search terms in text
function highlightSearchTerms(text, searchTerm) {
    if (!searchTerm) return text;
    
    const escapeRegExp = (string) => {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    };
    
    const regex = new RegExp(`(${escapeRegExp(searchTerm)})`, 'gi');
    return text.replace(regex, '<span class="highlighted">$1</span>');
}

// Initialize when the DOM is ready
document.addEventListener('DOMContentLoaded', initializeQA);